# Local macOS basketball app

The native dialogs run the Blender basketball renderer, COCO exporter, and offline
RGB/mask preview from the CLI. The build currently targets Apple Silicon.
Python and Pillow are bundled; Blender 5.x must be installed separately. Generation
requires no network or installed Python. This is a procedural 3D prototype;
photorealism and training benefit are not validated. The GUI offers quality
selection and a local progress page that automatically opens the finished gallery.

From the repository root, in a Python 3.12 environment with this project and
PyInstaller installed:

```sh
python -m PyInstaller --noconfirm macos/Generator.spec
"dist/Just a Sports Dataset Generator.app/Contents/MacOS/Just a Sports Dataset Generator" --demo /tmp/sports-demo
```

The demo renders three 3D images and must use a new output directory. The old
rectangle fixture is available with `--reference-demo`. Open `Vorschau/index.html` within its
output to inspect RGB images alongside colored mask previews. The GUI offers
“Bilder ansehen” on completion. Raw instance masks retain their numeric IDs and
may look black in ordinary image viewers.

The bundle uses local ad-hoc signing; distribution is not Apple-notarized. Include
the project, Python, Pillow/bundled image libraries, and bundled runtime libraries'
license notices with any redistributed package. Source for the launcher is
covered by the project's Apache-2.0 license. No models, telemetry, downloads, or
external image services are included.
