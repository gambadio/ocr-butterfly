# OCR Butterfly

**Mac-native OCR for Apple Silicon — images, PDFs, and video frames, all processed locally.**

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%203.0-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange.svg)](https://github.com/ml-explore/mlx)

OCR Butterfly uses the [DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR) model via Apple's [MLX framework](https://github.com/ml-explore/mlx) to run OCR entirely on your Mac. No cloud, no uploads, no subscriptions — just local GPU-accelerated text extraction with a clean web UI.

---

## Features

- **Image OCR** — Documents, tables, handwriting, scene text. Outputs Markdown, LaTeX, or plain text.
- **PDF Batch Processing** — Multi-page OCR with thumbnail preview, page selection, pause/resume controls.
- **Video Frame Extraction** — Extract key frames from MP4/AVI/MOV/MKV/WebM and batch-OCR them.
- **Image Preprocessing** — Auto-rotate, contrast enhancement, shadow removal, binarization, AI background removal.
- **100% Local** — All processing runs on-device via Metal GPU. The model downloads once (~800MB) and is cached locally.
- **Modern Web UI** — Dark cosmic theme, drag-and-drop uploads, real-time progress tracking.

## Requirements

| Requirement | Minimum |
|-------------|---------|
| **macOS** | 13.0+ |
| **Chip** | Apple Silicon (M1/M2/M3/M4) |
| **Python** | 3.11+ |
| **RAM** | 16 GB recommended |
| **Disk** | ~5 GB (including model cache) |

## Quick Start

```bash
# Clone the repo
git clone https://github.com/gambadio/ocr-butterfly.git
cd ocr-butterfly

# One-click start (creates venv, installs deps, launches app)
./start.sh
```

The app opens at `http://localhost:5001` (auto-selects from ports 5001-5010).

### Manual Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the Flask server
python3 app.py

# Or run as a native macOS window (pywebview)
python3 launcher.py
```

### First Run

On the first OCR request, the model (`mlx-community/DeepSeek-OCR-2-8bit`, ~800MB) downloads automatically from HuggingFace to `~/hf_cache/`. Subsequent runs use the cached model instantly.

## Architecture

Three source files:

| File | Purpose |
|------|---------|
| `app.py` | Flask backend — API endpoints, OCR pipeline, image/PDF/video processing |
| `static/app.js` | Frontend — file uploads, tab navigation, batch processing UI |
| `templates/index.html` | HTML structure and CSS (cosmic dark theme) |

**How OCR works:** Each OCR request spawns a subprocess that loads the model, runs inference via Metal GPU, and returns results. This isolates model crashes from the web server and enforces a 160-second timeout per image.

### API Endpoints

```
POST /api/ocr                    — Single image OCR
POST /api/pdf/{init,extract-pages,preview-page,process-batch,cancel}
POST /api/preprocess/{upload,process,download,to-ocr}
POST /api/video/{upload,extract,download,process-batch}
GET  /api/status                 — Model status
GET  /api/health                 — Server health
```

## Configuration

OCR quality is controlled by a 3D classification system:

- **Content Type**: Document or Scene
- **Subcategory**: Academic, Business, Table, Handwritten, Street, Photo, etc.
- **Complexity**: Tiny / Small / Medium / Large / Gundam

Higher complexity = larger image size (512-1280px) and more tokens (256-8192), at the cost of processing time.

## Differences from Upstream

OCR Butterfly is derived from [MLX-Video-OCR-DeepSeek-Apple-Silicon](https://github.com/matica0902/MLX-Video-OCR-DeepSeek-Apple-Silicon) by matica0902. Key differences:

- Full English UI and documentation (upstream was Traditional Chinese)
- Language-neutral OCR prompts (works with any language input)
- Removed packaging scripts — this repo is source-only for developers
- Modernized repo hygiene (`.editorconfig`, linter config, dependency audit)
- Professional English README with clear attribution

See [ATTRIBUTION.md](ATTRIBUTION.md) for full upstream credit.

## Privacy

All processing happens locally on your Mac:
- No data is sent to external servers
- Uploaded files are stored in system temp directories and auto-cleaned
- The model cache lives in `~/hf_cache/`
- No telemetry, no analytics, no cloud dependencies (after initial model download)

## License

**AGPL-3.0-or-later** — see [LICENSE](LICENSE).

This is a copyleft license. You can use, modify, and distribute this software freely, but any modified version must also be released under AGPL-3.0 with source code available. If you run a modified version as a network service, users must be able to obtain the source.

**Important:** PyMuPDF (used for PDF rendering) is also AGPL-3.0. All other dependencies use permissive licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the full audit.

## Attribution

This project is derived from [MLX-Video-OCR-DeepSeek-Apple-Silicon](https://github.com/matica0902/MLX-Video-OCR-DeepSeek-Apple-Silicon) by **matica0902**, licensed under AGPL-3.0. See [ATTRIBUTION.md](ATTRIBUTION.md) for details.
