import hashlib
import http.server
import io
import json
import os
import shutil
import sys
import tarfile
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

from just_a_sports_dataset_generator.generators.qwen import ensure_installed
from just_a_sports_dataset_generator.generators.qwen_setup import (
    LORA_FILE,
    PYTHON_BUILDS,
    SetupError,
    describe,
    install,
    installed,
    plan_install,
    python_triple,
)

BASE_FILES = {
    "model_index.json": b'{"_class_name": "QwenImageEditPlusPipeline"}',
    "transformer/config.json": b"{}",
    "text_encoder/model.safetensors": b"text-encoder-weights" * 50,
    "vae/diffusion_pytorch_model.safetensors": b"vae-weights" * 40,
    "README.md": b"not downloaded",
    "transformer/diffusion_pytorch_model.safetensors": b"skipped: GGUF replaces it",
}
REPOS = {
    "Qwen/Qwen-Image-Edit-2511": BASE_FILES,
    "unsloth/Qwen-Image-Edit-2511-GGUF": {
        "qwen-image-edit-2511-Q4_K_M.gguf": b"gguf" * 5000,
        "qwen-image-edit-2511-Q8_0.gguf": b"other quant",
    },
    "lightx2v/Qwen-Image-Edit-2511-Lightning": {LORA_FILE: b"lora" * 300},
}
LFS = {"text_encoder/model.safetensors", "qwen-image-edit-2511-Q4_K_M.gguf", LORA_FILE}


def python_archive(evil: bool = False) -> bytes:
    """A tiny stand-in for a python-build-standalone archive: python/bin/python3 runs us."""
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as bundle:
        script = f'#!/bin/sh\nexec "{sys.executable}" "$@"\n'.encode()
        info = tarfile.TarInfo("../evil" if evil else "python/bin/python3")
        info.size, info.mode = len(script), 0o755
        bundle.addfile(info, io.BytesIO(script))
    return buffer.getvalue()


class FakeHub(http.server.BaseHTTPRequestHandler):
    license = "apache-2.0"
    python: bytes = b""
    corrupt: str | None = None
    ranges: ClassVar[list[str]] = []

    def log_message(self, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        if self.path.startswith("/pbs/"):
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.python)))
            self.end_headers()
            self.wfile.write(self.python)
            return
        if self.path.startswith("/api/models/"):
            repo = self.path.removeprefix("/api/models/").split("/revision/")[0]
            self._json(self._info(repo))
            return
        repo_part, _, rest = self.path.lstrip("/").partition("/resolve/")
        _revision, _, name = rest.partition("/")
        data = REPOS[repo_part][name]
        if name == self.corrupt:
            data = b"X" + data[1:]
        start = 0
        if header := self.headers.get("Range"):
            FakeHub.ranges.append(name)
            start = int(header.removeprefix("bytes=").rstrip("-"))
        self.send_response(206 if start else 200)
        self.send_header("Content-Length", str(len(data) - start))
        self.end_headers()
        self.wfile.write(data[start:])

    def _info(self, repo: str) -> dict[str, object]:
        siblings = []
        for name, data in REPOS[repo].items():
            entry: dict[str, object] = {"rfilename": name, "size": len(data)}
            if name in LFS:
                entry["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            else:
                blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False)
                entry["blobId"] = blob.hexdigest()
            siblings.append(entry)
        sha = hashlib.sha1(repo.encode(), usedforsecurity=False).hexdigest()
        return {"sha": sha, "cardData": {"license": self.license}, "siblings": siblings}

    def _json(self, value: object) -> None:
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class QwenSetupTests(unittest.TestCase):
    def setUp(self) -> None:
        FakeHub.license, FakeHub.corrupt, FakeHub.ranges = "apache-2.0", None, []
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeHub)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.tmp = tempfile.TemporaryDirectory()
        self.home = Path(self.tmp.name) / "home"
        self.env = patch.dict(
            os.environ,
            {
                "SPORTS_DATASET_HF_ENDPOINT": f"http://127.0.0.1:{self.server.server_port}",
                "SPORTS_DATASET_HOME": str(self.home),
                "SPORTS_DATASET_PYTHON_BASE": f"http://127.0.0.1:{self.server.server_port}/pbs",
                "no_proxy": "127.0.0.1,localhost",
                "NO_PROXY": "127.0.0.1,localhost",
            },
        )
        self.env.start()

    def tearDown(self) -> None:
        self.env.stop()
        self.server.shutdown()
        self.tmp.cleanup()

    def plan(self):
        return replace(plan_install(), packages=())  # no PyPI in unit tests

    def test_plan_lists_exact_files_sizes_and_licenses(self):
        plan = plan_install()
        names = {item.path for component in plan.components for item in component.files}
        self.assertIn("text_encoder/model.safetensors", names)
        self.assertIn("qwen-image-edit-2511-Q4_K_M.gguf", names)
        self.assertNotIn("qwen-image-edit-2511-Q8_0.gguf", names)
        self.assertNotIn("README.md", names)
        self.assertNotIn("transformer/diffusion_pytorch_model.safetensors", names)
        text = describe(plan)
        for repo in REPOS:
            self.assertIn(repo, text)
        self.assertIn("apache-2.0", text)
        self.assertFalse(self.home.exists(), "planning must not write anything")

    def test_other_licenses_and_unknown_quants_are_refused(self):
        FakeHub.license = "other"
        with self.assertRaisesRegex(SetupError, "only apache-2.0"):
            plan_install()
        with self.assertRaises(SetupError):
            plan_install("Q1")

    def test_declined_consent_installs_nothing(self):
        asked = []
        with self.assertRaisesRegex(SetupError, "abgelehnt"):
            ensure_installed(lambda plan: asked.append(plan) is not None, progress=lambda _: None)
        self.assertEqual(len(asked), 1)
        self.assertFalse(self.home.exists())

    def test_install_verifies_every_file_and_writes_the_manifest_last(self):
        manifest_path = install(self.plan(), progress=lambda _: None, python=Path(sys.executable))
        manifest = installed()
        assert manifest is not None
        self.assertEqual(manifest_path, self.home / "qwen/installed.json")
        self.assertTrue(Path(str(manifest["gguf"])).read_bytes().startswith(b"gguf"))
        self.assertTrue(Path(str(manifest["lora"])).name == LORA_FILE)
        base = Path(str(manifest["base_dir"]))
        self.assertEqual(
            (base / "vae/diffusion_pytorch_model.safetensors").read_bytes()[:3], b"vae"
        )
        self.assertFalse((base / "README.md").exists())
        self.assertFalse((self.home / "qwen/downloads").exists())
        # Already installed: no second consent request.
        self.assertEqual(ensure_installed(lambda plan: False), manifest)

    def test_checksum_mismatch_aborts_without_a_manifest(self):
        FakeHub.corrupt = "qwen-image-edit-2511-Q4_K_M.gguf"
        with self.assertRaisesRegex(SetupError, "checksum mismatch"):
            install(self.plan(), progress=lambda _: None, python=Path(sys.executable))
        self.assertIsNone(installed())
        self.assertFalse(list((self.home / "qwen/downloads").rglob("*.part")))
        # A verified file is never downloaded again, so start from empty models.
        shutil.rmtree(self.home / "qwen/models")
        FakeHub.corrupt = "model_index.json"  # git blob checksum of a small file
        with self.assertRaisesRegex(SetupError, "checksum mismatch"):
            install(self.plan(), progress=lambda _: None, python=Path(sys.executable))

    def test_interrupted_downloads_resume_with_a_range_request(self):
        plan = self.plan()
        name = "qwen-image-edit-2511-Q4_K_M.gguf"
        data = REPOS["unsloth/Qwen-Image-Edit-2511-GGUF"][name]
        partial = self.home / "qwen/downloads/unsloth--Qwen-Image-Edit-2511-GGUF" / (name + ".part")
        partial.parent.mkdir(parents=True)
        partial.write_bytes(data[:7000])
        install(plan, progress=lambda _: None, python=Path(sys.executable))
        self.assertEqual(FakeHub.ranges, [name])
        manifest = installed()
        assert manifest is not None
        self.assertEqual(Path(str(manifest["gguf"])).read_bytes(), data)

    @unittest.skipIf(sys.platform == "win32", "the stand-in interpreter is a POSIX script")
    def test_portable_python_is_downloaded_verified_and_used(self):
        archive = python_archive()
        FakeHub.python = archive
        triple = python_triple()
        if triple is None:
            self.skipTest("no pinned portable Python for this machine")
        digest = hashlib.sha256(archive).hexdigest()
        with patch.dict(PYTHON_BUILDS, {triple: digest}):
            plan = replace(plan_install(), packages=())
            assert plan.python is not None
            self.assertIn("python-build-standalone", describe(plan))
            install(plan, progress=lambda _: None)
            manifest = installed()
            assert manifest is not None
            self.assertEqual(manifest["portable_python"]["sha256"], digest)  # type: ignore[index]
            self.assertTrue((self.home / "qwen/python/bin/python3").is_file())
            self.assertTrue(Path(str(manifest["python"])).is_file())

    def test_portable_python_checksum_and_paths_are_enforced(self):
        triple = python_triple()
        if triple is None:
            self.skipTest("no pinned portable Python for this machine")
        FakeHub.python = python_archive()
        wrong = patch.dict(PYTHON_BUILDS, {triple: "0" * 64})
        with wrong, self.assertRaisesRegex(SetupError, "checksum mismatch for the Python"):
            install(replace(plan_install(), packages=()), progress=lambda _: None)
        evil = python_archive(evil=True)
        FakeHub.python = evil
        matching = patch.dict(PYTHON_BUILDS, {triple: hashlib.sha256(evil).hexdigest()})
        with matching, self.assertRaisesRegex(SetupError, "unsafe path"):
            install(replace(plan_install(), packages=()), progress=lambda _: None)
        self.assertFalse((self.home.parent / "evil").exists())
        self.assertIsNone(installed())


if __name__ == "__main__":
    unittest.main()
