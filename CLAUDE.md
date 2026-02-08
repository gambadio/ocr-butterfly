# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OCR Butterfly is a Mac-native OCR application for Apple Silicon. It uses the DeepSeek-OCR-2-8bit model via the MLX framework to perform OCR on images, PDFs, and video frames. The app runs entirely locally with a Flask backend and vanilla JS frontend, optionally wrapped in a native macOS window via pywebview.

Derived from [MLX-Video-OCR-DeepSeek-Apple-Silicon](https://github.com/matica0902/MLX-Video-OCR-DeepSeek-Apple-Silicon) by matica0902 (AGPL-3.0).

## Running the Application

```bash
# Activate virtual environment
source venv/bin/activate

# Run Flask server directly
python3 app.py

# Run as native macOS app (pywebview window with loading screen)
python3 launcher.py

# One-click start (auto venv + deps + launch)
./start.sh
```

The server listens on `127.0.0.1` and auto-selects a port from 5001-5010. The model (~800MB) downloads from HuggingFace on first OCR request to `~/hf_cache/`.

## Dependencies

```bash
pip install -r requirements.txt
```

Python 3.11+ required. Requires Apple Silicon (M1+) for Metal GPU acceleration. Key deps: Flask 3.0, mlx-vlm >= 0.3.11, mlx >= 0.20.0, PyMuPDF, opencv-python, pywebview, transformers >= 5.1.0. Optional: `rembg` for background removal (graceful fallback if missing).

## Architecture

The entire application is three source files:

- **app.py** — Flask backend with all API endpoints, OCR pipeline, image preprocessing, PDF/video processing
- **static/app.js** — Frontend logic: file uploads, tab navigation, batch processing UI, model status polling
- **templates/index.html** — HTML structure and CSS styles (cosmic dark theme)

### Key Architectural Patterns

**Subprocess OCR execution**: OCR runs in a spawned subprocess (`multiprocessing`) to isolate model crashes from the Flask server, enforce timeouts (160s), and enable GPU cleanup. The model loads lazily on first OCR request inside the subprocess.

**Task-based async operations**: PDF batch processing, image preprocessing, and video frame extraction use UUID-tracked tasks stored in `pdf_tasks`, `preprocess_tasks`, and `video_tasks` dicts. Tasks auto-expire after 30 minutes.

**Configuration grid**: A 3D classification system maps (content_type, subcategory, complexity) to processing parameters. Two content types (Document, Scene) with 6+4 subcategories and 5 complexity levels (Tiny/Small/Medium/Large/Gundam) determine `image_size` (512-1280px) and `max_tokens` (256-8192). This is the `PREPROCESSING_CONFIG` dict at the top of app.py.

**Launcher pattern**: `launcher.py` shows a native pywebview window immediately (so macOS sees a GUI app), starts Flask in a background thread, and navigates to the app once the server is ready. Closing the window kills Flask.

### API Surface

Core endpoints follow a resource pattern:
- `/api/ocr` - Single image OCR
- `/api/pdf/{init,extract-pages,preview-page,process-batch,cancel}` - PDF pipeline
- `/api/preprocess/{upload,process,download,to-ocr}` - Image preprocessing pipeline
- `/api/video/{upload,extract,download,process-batch}` - Video frame pipeline
- `/api/status`, `/api/health` - System status

## Important Constants (app.py)

- `MODEL_NAME = "mlx-community/DeepSeek-OCR-2-8bit"`
- `ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}`
- `ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}`
- `MAX_CONTENT_LENGTH = 512MB`
- OCR subprocess timeout: 160 seconds
- Task expiration: 30 minutes

## License

AGPL-3.0-or-later. All source files include SPDX license headers with upstream attribution.
