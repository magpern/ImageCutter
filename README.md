# Image Split

Desktop tool for splitting an image into a grid of tiles using vertical and horizontal cut lines, with optional auto-detection of white gaps (contact-sheet style), per-tile export formats, and extras like round-icon extraction.

## Requirements

- **Python 3.10+** (recommended; needs a standard library that includes `tkinter`)
- **Windows / macOS / Linux** with a GUI (Tk needs a display)

## Install

```bash
cd ImageSplit
python -m venv .venv
```

Activate the venv (examples):

- Windows (PowerShell): `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

Then:

```bash
pip install -r requirements.txt
```

Dependencies:

| Package | Purpose |
|--------|---------|
| [Pillow](https://python-pillow.org/) | Image I/O, resize, export |
| [NumPy](https://numpy.org/) | Auto grid (white-gap) detection |
| [opencv-python-headless](https://pypi.org/project/opencv-python-headless/) | Circle detection in cells (optional at runtime if missing) |

## Run

```bash
python app.py
```

If you use **VS Code**, you can start debugging from `.vscode/launch.json`.

## Standalone binaries (CI)

The workflow **Build binaries** (`.github/workflows/build-binaries.yml`) runs on **workflow_dispatch** (Actions tab → run workflow) and on **pushes to tags** matching `v*`. It builds a **PyInstaller** one-file app on **Windows**, **macOS**, and **Ubuntu**, and uploads three **workflow artifacts** (one per OS). Download an artifact for your platform, extract if needed, and run the executable. macOS builds are **not** notarized; you may need to allow the app in Security settings the first time.

### Adding binaries to a GitHub Release

**Automatic (recommended):** Create the tag **first** (`git tag v1.0.1 && git push origin v1.0.1`). The workflow builds all platforms, then the **Attach to GitHub Release** job uploads `ImageCutter-<tag>-windows-amd64.exe`, `ImageCutter-<tag>-macos`, and `ImageCutter-<tag>-linux-amd64` to the release for that tag (creating the release if it does not exist yet). If you already published `v1.0.0` from the GitHub UI **before** this job existed, either **re-run** the latest successful **Build binaries** run for that tag (if listed under Actions), or use the manual steps below.

**Manual:** Open the successful workflow run → download each artifact ZIP → unzip → on the release page click **Edit** → drag the files into **Attach binaries** (or use **Add assets**).

## What you can do

- **Open** an image, **add vertical / horizontal lines** (or **move** / **delete** them).
- **Zoom** (wheel and toolbar), **pan** with scrollbars when zoomed, **fit to window**.
- **Export tiles** to a folder as **PNG**, **JPEG**, or **WebP**, with **quality / compression** options (remembered per format).
- **Table (white gap) grid**: choose **rows × columns** (1–10 each), then **Auto grid (gaps)…** to detect full-span near-white gutters and place lines so export skips the gaps (montage / contact-sheet layout).
- **Export: round cutout (transparent)**: batch-export each tile as a circular cutout with transparency where supported (JPEG is upgraded to PNG when transparency is required).
- **Extract circle (click cell)**: single-tile PNG with a soft circular alpha mask (requires OpenCV).
- **Ignore tile (click cell)**: mark tiles you do not want exported; they are hatched in the preview and skipped on export. **Tools → Clear ignored tiles** removes marks only.

Keyboard shortcuts (when the app has focus):

- **Ctrl+O** — Open image  
- **Ctrl+E** — Export tiles  
- **Ctrl+Q** — Exit  

## Project layout

| File | Role |
|------|------|
| `app.py` | Tk UI, lines, zoom, export, montage, ignore tiles |
| `circle_extract.py` | OpenCV-based circle fit and RGBA crop |
| `requirements.txt` | Python dependencies |

## Notes

- **Circle** features need a working OpenCV install (`CIRCLE_EXTRACT_OK`). If import fails, the rest of the app still runs; only circle extract / transparent circle batch paths are affected.
- **Ignore** marks are cleared when you **clear lines** or **open a new image**; marks that no longer match the grid after you edit lines are dropped automatically.

## License

No license file is included in this repository; add one if you distribute the project.
