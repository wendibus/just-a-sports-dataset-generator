# Local macOS basketball app

The native dialogs run the Blender basketball renderer, COCO exporter, and offline
RGB/mask preview from the CLI. The build currently targets Apple Silicon.
Python and Pillow are bundled; Blender 5.x must be installed separately. Generation
requires no network or installed Python. This is a procedural 3D prototype;
photorealism and training benefit are not validated. The GUI offers quality
selection and a local progress page that automatically opens the finished gallery.

## Photorealistic players (optional)

After the quality choice the app asks whether to repaint the players with a local
Qwen-Image-Edit-2511. If Qwen is not installed, a native dialog lists every download
(files, sizes, licenses, destination, about 31 GB including a portable Python) and
installs only after “Herunterladen”; installation progress appears on the progress
page. No system Python, Homebrew or Terminal is needed. Labels stay unchanged; see
[pose crops](../docs/POSE_CROPS.md).

## Cancel and resume

While a run is busy, a second dialog offers **Abbrechen**. It stops the running Blender
or Qwen process; finished scenes and generated players are kept in
`Ausgaben/Durchlauf-….partial`. On the next start the app offers **Fortsetzen**, which
renders and generates only what is missing and then publishes the run as
`Durchlauf-…`. “Neu starten” leaves the unfinished folder in place for manual deletion.
The same resume behaviour applies after a crash or power loss. The last unfinished run
is remembered in `launcher-state.json` in the user data folder.

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
covered by the project's Apache-2.0 license. No models, telemetry, or external image services are included; the optional Qwen
download happens only after the consent dialog.
