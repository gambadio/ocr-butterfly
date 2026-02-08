# Third-Party Notices

OCR Butterfly depends on the following open-source components.
Each is listed with its license type and compatibility notes for AGPL-3.0.

## Runtime Dependencies

| Component | License | Compatible | Notes |
|-----------|---------|------------|-------|
| **Flask** 3.0 | BSD-3-Clause | Yes | Web framework |
| **Werkzeug** 3.0 | BSD-3-Clause | Yes | WSGI toolkit (Flask dependency) |
| **mlx** >= 0.20.0 | MIT | Yes | Apple ML framework |
| **mlx-vlm** >= 0.3.11 | MIT | Yes | MLX vision-language models |
| **Pillow** >= 10.3.0 | MIT-CMU (HPND) | Yes | Image processing |
| **PyMuPDF** (fitz) | AGPL-3.0 | Yes | PDF rendering. Same license family. |
| **opencv-python** >= 4.10.0 | Apache-2.0 | Yes | Computer vision |
| **pywebview** >= 5.0 | BSD-3-Clause | Yes | Native desktop window |
| **transformers** >= 5.1.0 | Apache-2.0 | Yes | HuggingFace model loading |

## Optional Dependencies

| Component | License | Compatible | Notes |
|-----------|---------|------------|-------|
| **rembg** | MIT | Yes | AI background removal (graceful fallback) |

## Model

| Component | License | Notes |
|-----------|---------|-------|
| **DeepSeek-OCR** | Model license per DeepSeek | Downloaded at runtime from HuggingFace, not bundled |
| **mlx-community/DeepSeek-OCR-2-8bit** | Same as above | 8-bit quantized variant |

## License Flags

- **PyMuPDF (AGPL-3.0)**: This is a "viral" copyleft license, same as this project.
  Since OCR Butterfly is itself AGPL-3.0, there is no conflict. However, any
  derivative work must also be AGPL-3.0.
- **All other dependencies** use permissive licenses (MIT, BSD, Apache-2.0) that
  are fully compatible with AGPL-3.0.
- **DeepSeek-OCR model**: Downloaded at runtime by the user. Not distributed
  with this source code. Check the model's license on HuggingFace for your
  specific use case.
