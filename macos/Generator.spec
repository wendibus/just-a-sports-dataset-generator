"""Build the standalone Apple Silicon reference app with PyInstaller."""
from pathlib import Path
import tomllib

root = Path(SPECPATH).parent
version = tomllib.loads((root / "pyproject.toml").read_text())["project"]["version"]
a = Analysis(
    [str(root / "macos/launcher.py")],
    pathex=[str(root / "src")],
    datas=[(str(root / "examples"), "examples"),
           (str(root / "licenses"), "Lizenzen"),
           (str(root / "LICENSE"), "Lizenzen"),
           (str(root / "src/just_a_sports_dataset_generator/renderers/blender_worker/basketball.py"),
            "just_a_sports_dataset_generator/renderers/blender_worker")],
    excludes=["numpy", "tkinter", "pycocotools"],
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name="Just a Sports Dataset Generator", console=False, target_arch="arm64",
)
coll = COLLECT(exe, a.binaries, a.datas, name="Just a Sports Dataset Generator")
app = BUNDLE(
    coll, name="Just a Sports Dataset Generator.app",
    bundle_identifier="org.wendibus.sports-dataset-generator", version=version,
    info_plist={"CFBundleVersion": version, "NSHumanReadableCopyright": "Apache-2.0 — wendibus"},
)
