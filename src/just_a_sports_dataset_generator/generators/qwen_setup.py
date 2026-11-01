"""Consent-based local installation of the Qwen-Image-Edit-2511 runtime.

Nothing is downloaded silently. :func:`plan_install` asks the Hugging Face API for the exact
files, sizes, licenses and commit of every component; the caller shows that plan and only
:func:`install` with an approved plan touches the network for large downloads. All weights
are Apache-2.0; a component reporting another license is refused.

Layout under :func:`data_home` (outside the application, kept across updates)::

    qwen/runtime/     isolated Python environment with pinned packages
    qwen/models/      verified model files, one folder per repository commit
    qwen/downloads/   resumable partial downloads
    qwen/installed.json  written last; its presence means a complete, verified install
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import asdict, dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import cast

from just_a_sports_dataset_generator.storage import write_json

LICENSE = "apache-2.0"
QUANTS = ("Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
LORA_FILE = "Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors"
# Versions reported working together for GGUF Qwen-Image-Edit-2511 in diffusers (late 2025).
PACKAGES = (
    "torch==2.9.1",
    # Qwen2.5-VL's processor loads a video processor that needs torchvision; 0.24.1 is the
    # release built against torch 2.9.1 (it pins torch==2.9.1).
    "torchvision==0.24.1",
    "diffusers==0.36.0",
    "transformers==4.57.3",
    "accelerate==1.12.0",
    "peft==0.18.0",
    "gguf==0.17.1",
    "safetensors==0.7.0",
    "huggingface-hub==0.36.0",
    "Pillow>=11.3",
)
PACKAGES_ESTIMATE_BYTES = 2_000_000_000
# Portable CPython from python-build-standalone, pinned by release, version and SHA-256, so
# the runtime never depends on (or alters) a system Python. Checksums from the release's
# SHA256SUMS; the Linux x86_64 build was downloaded and verified when pinning.
PYTHON_RELEASE, PYTHON_VERSION = "20260901", "3.12.14"
PYTHON_BUILDS = {
    "aarch64-apple-darwin": "81a359f1cfadd4da11766534c5913791cea55f26e1bb902cacd2a531bb1e4b2b",
    "x86_64-apple-darwin": "65b195c9cedc1fef6767f044f9822069adbd1bd9204d424ece4628776fdc04bb",
    "aarch64-unknown-linux-gnu": "577b4bec0793ad1ff0cbff9adbd0df078eddde38a4c41bf5d83ad381a85ee39d",
    "x86_64-unknown-linux-gnu": "72748da13197c1fb161e3afeef20a6a385ff24f2165e6e2758e47008e7faba4c",
    "x86_64-pc-windows-msvc": "7c45c9622400d578709a9b2cddbe8124cc21d382409d9f13406d706d28e31b14",
}
PYTHON_ESTIMATE_BYTES = 35_000_000
INSTALL_SCHEMA = "1.0"


def hf_endpoint() -> str:
    """Hugging Face base URL; overridable for mirrors and tests."""
    return os.environ.get("SPORTS_DATASET_HF_ENDPOINT", "https://huggingface.co").rstrip("/")


class SetupError(RuntimeError):
    """Raised when the Qwen runtime cannot be planned, approved or installed."""


@dataclass(frozen=True, slots=True)
class PlannedFile:
    repo: str
    revision: str
    path: str
    size: int
    sha256: str | None
    git_sha1: str | None


@dataclass(frozen=True, slots=True)
class Component:
    repo: str
    revision: str
    license: str
    purpose: str
    files: tuple[PlannedFile, ...]

    @property
    def size(self) -> int:
        return sum(item.size for item in self.files)


@dataclass(frozen=True, slots=True)
class PortablePython:
    triple: str
    url: str
    sha256: str
    version: str


@dataclass(frozen=True, slots=True)
class InstallPlan:
    home: Path
    quant: str
    components: tuple[Component, ...]
    packages: tuple[str, ...]
    python: PortablePython | None = None
    """Portable interpreter to download; None falls back to a local Python 3.10–3.13."""

    @property
    def download_bytes(self) -> int:
        models = sum(component.size for component in self.components)
        return models + (PYTHON_ESTIMATE_BYTES if self.python else 0)


def python_triple() -> str | None:
    """python-build-standalone target for this machine, or None if no pinned build fits."""
    machine = platform.machine().lower()
    arch = {"arm64": "aarch64", "aarch64": "aarch64", "x86_64": "x86_64", "amd64": "x86_64"}.get(
        machine
    )
    system = {"darwin": "apple-darwin", "linux": "unknown-linux-gnu", "win32": "pc-windows-msvc"}
    if arch is None or sys.platform not in system:
        return None
    triple = f"{arch}-{system[sys.platform]}"
    return triple if triple in PYTHON_BUILDS else None


def portable_python() -> PortablePython | None:
    triple = python_triple()
    if triple is None:
        return None
    base = os.environ.get(
        "SPORTS_DATASET_PYTHON_BASE",
        "https://github.com/astral-sh/python-build-standalone/releases/download",
    ).rstrip("/")
    name = f"cpython-{PYTHON_VERSION}%2B{PYTHON_RELEASE}-{triple}-install_only_stripped.tar.gz"
    return PortablePython(
        triple, f"{base}/{PYTHON_RELEASE}/{name}", PYTHON_BUILDS[triple], PYTHON_VERSION
    )


def data_home() -> Path:
    override = os.environ.get("SPORTS_DATASET_HOME")
    if override:
        return Path(override)
    if sys.platform == "darwin":
        return Path.home() / "Library/Application Support/JustASportsDatasetGenerator"
    if sys.platform == "win32":
        return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "JustASportsDatasetGenerator"
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local/share")
    return Path(base) / "just-a-sports-dataset-generator"


def qwen_home(home: Path | None = None) -> Path:
    return (home or data_home()) / "qwen"


def runtime_python(home: Path | None = None) -> Path:
    runtime = qwen_home(home) / "runtime"
    return runtime / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _api(repo: str, revision: str) -> dict[str, object]:
    url = (
        f"{hf_endpoint()}/api/models/{repo}/revision/"
        f"{urllib.parse.quote(revision, safe='')}?blobs=true"
    )
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data: object = json.loads(response.read(8 * 1024 * 1024))
    except OSError as exc:
        raise SetupError(f"could not reach Hugging Face for {repo}: {exc}") from exc
    if not isinstance(data, dict):
        raise SetupError(f"unexpected Hugging Face response for {repo}")
    return cast(dict[str, object], data)


def _license(info: dict[str, object]) -> str:
    card = info.get("cardData")
    if isinstance(card, dict):
        value = cast(dict[str, object], card).get("license")
        if isinstance(value, str):
            return value.lower()
    for tag in cast(list[object], info.get("tags") or []):
        if isinstance(tag, str) and tag.startswith("license:"):
            return tag.removeprefix("license:").lower()
    return "unknown"


def _component(
    repo: str, purpose: str, select: Callable[[str], bool], required: tuple[str, ...] = ()
) -> Component:
    info = _api(repo, "main")
    revision = info.get("sha")
    if not isinstance(revision, str) or len(revision) != 40:
        raise SetupError(f"{repo} did not report a commit")
    license_id = _license(info)
    if license_id != LICENSE:
        raise SetupError(f"{repo} reports license {license_id!r}; only {LICENSE} is accepted")
    files: list[PlannedFile] = []
    for raw in cast(list[object], info.get("siblings") or []):
        if not isinstance(raw, dict):
            continue
        sibling = cast(dict[str, object], raw)
        name = sibling.get("rfilename")
        if not isinstance(name, str) or not select(name):
            continue
        if name.startswith("/") or ".." in Path(name).parts or "\\" in name:
            raise SetupError(f"{repo} lists an unsafe file name: {name!r}")
        lfs = sibling.get("lfs")
        sha256 = cast(dict[str, object], lfs).get("sha256") if isinstance(lfs, dict) else None
        size = sibling.get("size")
        blob = sibling.get("blobId")
        if type(size) is not int or (not isinstance(sha256, str) and not isinstance(blob, str)):
            raise SetupError(f"{repo} has no size or checksum for {name}")
        files.append(
            PlannedFile(
                repo,
                revision,
                name,
                size,
                sha256 if isinstance(sha256, str) else None,
                None if isinstance(sha256, str) else cast(str, blob),
            )
        )
    names = {item.path for item in files}
    missing = [pattern for pattern in required if not any(fnmatch(n, pattern) for n in names)]
    if missing or not files:
        raise SetupError(f"{repo} is missing expected files: {', '.join(missing) or 'all'}")
    return Component(
        repo, revision, license_id, purpose, tuple(sorted(files, key=lambda f: f.path))
    )


def plan_install(quant: str = "Q4_K_M", home: Path | None = None) -> InstallPlan:
    """Resolve exact files, sizes, checksums and licenses. Downloads nothing large."""
    if quant not in QUANTS:
        raise SetupError(f"quantization must be one of: {', '.join(QUANTS)}")
    base_patterns = (
        "model_index.json",
        "scheduler/*",
        "text_encoder/*",
        "tokenizer/*",
        "processor/*",
        "vae/*",
        "transformer/config.json",
    )
    components = (
        _component(
            "Qwen/Qwen-Image-Edit-2511",
            "text encoder, VAE, tokenizer and configuration",
            lambda name: any(fnmatch(name, pattern) for pattern in base_patterns),
            ("model_index.json", "text_encoder/*", "vae/*", "transformer/config.json"),
        ),
        _component(
            "unsloth/Qwen-Image-Edit-2511-GGUF",
            f"image transformer, {quant} quantization",
            lambda name: name.endswith(f"-{quant}.gguf") and "/" not in name,
            (f"*-{quant}.gguf",),
        ),
        _component(
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "4-step Lightning LoRA",
            lambda name: name == LORA_FILE,
            (LORA_FILE,),
        ),
    )
    gguf = [f for f in components[1].files if f.path.endswith(".gguf")]
    if len(gguf) != 1:
        raise SetupError(f"expected exactly one {quant} GGUF file, found {len(gguf)}")
    return InstallPlan(qwen_home(home), quant, components, PACKAGES, portable_python())


def describe(plan: InstallPlan) -> str:
    """Human-readable consent text: sources, sizes, licenses, destination."""
    gigabytes = 1_000_000_000
    lines = [
        "Qwen-Image-Edit-2511 wird lokal eingerichtet (einmalig, danach offline nutzbar).",
        "",
    ]
    for component in plan.components:
        lines.append(
            f"- {component.repo} @ {component.revision[:12]} — {component.purpose}: "
            f"{len(component.files)} Datei(en), {component.size / gigabytes:.1f} GB, "
            f"Lizenz {component.license}"
        )
    if plan.python:
        lines.append(
            f"- Python {plan.python.version} (python-build-standalone {PYTHON_RELEASE}, "
            f"{plan.python.triple}): ca. {PYTHON_ESTIMATE_BYTES / 1e6:.0f} MB, Lizenz PSF-2.0"
        )
    lines += [
        (
            f"- Python-Pakete von PyPI (PyTorch, diffusers …): ca. "
            f"{PACKAGES_ESTIMATE_BYTES / gigabytes:.0f} GB"
        ),
        "",
        f"Download gesamt: ca. {(plan.download_bytes + PACKAGES_ESTIMATE_BYTES) / gigabytes:.1f} GB",
        f"Ziel: {plan.home}",
        "Speicherbedarf beim Generieren: ca. 30 GB Arbeitsspeicher.",
    ]
    return "\n".join(lines)


def find_python() -> Path:
    """A Python 3.10–3.13 interpreter to create the isolated runtime from."""
    if not getattr(sys, "frozen", False) and (3, 10) <= sys.version_info[:2] <= (3, 13):
        return Path(sys.executable)
    names = ("python3.13", "python3.12", "python3.11", "python3.10", "python3")
    folders = [Path("/opt/homebrew/bin"), Path("/usr/local/bin")]
    candidates = [Path(found) for name in names if (found := shutil.which(name))]
    candidates += [folder / name for folder in folders for name in names]
    for candidate in candidates:
        if not candidate.is_file():
            continue
        try:
            version = subprocess.run(
                [
                    str(candidate),
                    "-c",
                    (
                        "import sys; print(sys.version_info[:2] >= (3, 10) "
                        "and sys.version_info[:2] <= (3, 13))"
                    ),
                ],
                capture_output=True,
                text=True,
                timeout=20,
                check=False,
            ).stdout.strip()
        except OSError:
            continue
        if version == "True":
            return candidate
    raise SetupError("Python 3.10–3.13 wurde nicht gefunden; bitte installieren (z. B. python.org)")


def _verify(path: Path, item: PlannedFile) -> bool:
    if path.stat().st_size != item.size:
        return False
    if item.sha256 is not None:
        digest = hashlib.sha256()
    else:
        digest = hashlib.sha1(f"blob {item.size}\0".encode(), usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == (item.sha256 or item.git_sha1)


def _download(
    item: PlannedFile, target: Path, partial: Path, progress: Callable[[str], None]
) -> None:
    if target.is_file() and _verify(target, item):
        return
    partial.parent.mkdir(parents=True, exist_ok=True)
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > item.size:
        partial.unlink()
        offset = 0
    url = f"{hf_endpoint()}/{item.repo}/resolve/{item.revision}/{urllib.parse.quote(item.path)}"
    request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
    if offset < item.size:
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    offset = 0  # server ignored the range; start over
                with partial.open("ab" if offset else "wb") as stream:
                    done = offset
                    reported = -1
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
                        done += len(block)
                        percent = done * 100 // max(item.size, 1)
                        if percent // 10 != reported:
                            reported = percent // 10
                            progress(f"{item.path}: {percent} %")
        except OSError as exc:
            raise SetupError(f"download of {item.path} failed (resumable): {exc}") from exc
    if not _verify(partial, item):
        partial.unlink()
        raise SetupError(f"checksum mismatch for {item.path}; the partial file was removed")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial.replace(target)


def _pip_install(interpreter: Path, packages: tuple[str, ...], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        result = subprocess.run(
            [
                str(interpreter),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                # Slow or flaky connections: wait longer and retry instead of failing.
                "--timeout",
                "120",
                "--retries",
                "10",
                *packages,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise SetupError(f"package installation failed; see {log_path}")


def update_packages(home: Path | None, progress: Callable[[str], None]) -> dict[str, object]:
    """Bring an installed runtime's Python packages up to the pinned list.

    Models stay untouched; only PyPI packages are installed (small compared with the
    approved download). The manifest is rewritten once pip succeeded.
    """
    manifest = installed(home)
    if manifest is None:
        raise SetupError("Qwen is not installed")
    if manifest.get("packages") == list(PACKAGES):
        return manifest
    progress("Python-Pakete der Qwen-Umgebung werden aktualisiert …")
    folder = qwen_home(home)
    _pip_install(Path(str(manifest["python"])), PACKAGES, folder / "pip-update.log")
    manifest["packages"] = list(PACKAGES)
    write_json(folder / "installed.json", manifest)
    return manifest


def _install_python(python: PortablePython, home: Path, progress: Callable[[str], None]) -> Path:
    """Download, verify and safely unpack the portable interpreter; return its executable."""
    folder = home / "python"
    executable = folder / ("python.exe" if sys.platform == "win32" else "bin/python3")
    marker = folder / ".sha256"
    if executable.is_file() and marker.is_file() and marker.read_text().strip() == python.sha256:
        return executable
    progress(f"Python {python.version} wird geladen …")
    archive = home / "downloads" / "python.tar.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(python.url, timeout=60) as response, archive.open("wb") as out:
            shutil.copyfileobj(response, out, 4 * 1024 * 1024)
    except OSError as exc:
        raise SetupError(f"download of Python failed: {exc}") from exc
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    if digest.hexdigest() != python.sha256:
        archive.unlink()
        raise SetupError("checksum mismatch for the Python download; it was removed")
    staging = home / "python.partial"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    with tarfile.open(archive) as bundle:
        for member in bundle.getmembers():
            target = (staging / member.name).resolve()
            if not target.is_relative_to(staging.resolve()) or member.isdev():
                shutil.rmtree(staging, ignore_errors=True)
                raise SetupError(f"unsafe path in the Python archive: {member.name}")
        if hasattr(tarfile, "data_filter"):  # Python >= 3.11.4: also rejects unsafe links
            bundle.extractall(staging, filter="data")
        else:  # members were validated above
            bundle.extractall(staging)
    unpacked = staging / "python"
    if not (unpacked / executable.relative_to(folder)).is_file():
        shutil.rmtree(staging, ignore_errors=True)
        raise SetupError("the Python archive does not contain the expected interpreter")
    shutil.rmtree(folder, ignore_errors=True)
    unpacked.replace(folder)
    shutil.rmtree(staging, ignore_errors=True)
    archive.unlink()
    marker.write_text(python.sha256)
    return executable


def model_dir(home: Path, repo: str, revision: str) -> Path:
    return home / "models" / repo.replace("/", "--") / revision


def install(
    plan: InstallPlan,
    progress: Callable[[str], None] = print,
    python: Path | None = None,
) -> Path:
    """Create the runtime, download and verify every planned file, then write the manifest."""
    home = plan.home
    home.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(home).free
    packages = PACKAGES_ESTIMATE_BYTES if plan.packages else 0
    needed = plan.download_bytes + packages + 1_000_000_000
    if free < needed:
        raise SetupError(
            f"zu wenig Speicherplatz: {needed / 1e9:.0f} GB nötig, {free / 1e9:.0f} GB frei"
        )
    (home / "installed.json").unlink(missing_ok=True)
    runtime = home / "runtime"
    interpreter = runtime_python(home.parent)
    if not interpreter.is_file():
        if python is not None:
            base = python
        elif plan.python is not None:
            base = _install_python(plan.python, home, progress)
        else:
            base = find_python()
        progress("Python-Umgebung wird angelegt …")
        command = [str(base), "-m", "venv", str(runtime)]
        result = subprocess.run(
            command + ([] if plan.packages else ["--without-pip"]),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode or not interpreter.is_file():
            shutil.rmtree(runtime, ignore_errors=True)
            raise SetupError(f"could not create the Python environment: {result.stderr.strip()}")
    if plan.packages:
        progress("Python-Pakete werden installiert …")
        _pip_install(interpreter, plan.packages, home / "pip.log")
    for component in plan.components:
        folder = model_dir(home, component.repo, component.revision)
        for item in component.files:
            partial = home / "downloads" / component.repo.replace("/", "--") / (item.path + ".part")
            _download(item, folder / item.path, partial, progress)
    gguf = next(f for f in plan.components[1].files if f.path.endswith(".gguf"))
    manifest = {
        "schema_version": INSTALL_SCHEMA,
        "model": "Qwen-Image-Edit-2511",
        "quant": plan.quant,
        "packages": list(plan.packages),
        "python": str(interpreter),
        "portable_python": asdict(plan.python) if plan.python else None,
        "base_dir": str(model_dir(home, plan.components[0].repo, plan.components[0].revision)),
        "gguf": str(model_dir(home, gguf.repo, gguf.revision) / gguf.path),
        "lora": str(
            model_dir(home, plan.components[2].repo, plan.components[2].revision) / LORA_FILE
        ),
        "components": [
            {
                **{k: v for k, v in asdict(c).items() if k != "files"},
                "files": [
                    {"path": f.path, "size": f.size, "sha256": f.sha256, "git_sha1": f.git_sha1}
                    for f in c.files
                ],
            }
            for c in plan.components
        ],
    }
    write_json(home / "installed.json", manifest)
    shutil.rmtree(home / "downloads", ignore_errors=True)
    progress("Qwen ist eingerichtet.")
    return home / "installed.json"


def installed(home: Path | None = None) -> dict[str, object] | None:
    """The install manifest when every referenced file is present with its recorded size."""
    path = qwen_home(home) / "installed.json"
    if not path.is_file():
        return None
    try:
        data: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    manifest = cast(dict[str, object], data)
    if manifest.get("schema_version") != INSTALL_SCHEMA:
        return None
    for key in ("python", "gguf", "lora"):
        value = manifest.get(key)
        if not isinstance(value, str) or not Path(value).is_file():
            return None
    base = manifest.get("base_dir")
    if not isinstance(base, str) or not (Path(base) / "model_index.json").is_file():
        return None
    return manifest
