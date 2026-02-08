# Changelog

All notable changes to OCR Butterfly are documented here.

## [1.0.0] - 2026-02-08

### Initial Independent Release

Forked from [MLX-Video-OCR-DeepSeek-Apple-Silicon](https://github.com/matica0902/MLX-Video-OCR-DeepSeek-Apple-Silicon)
by matica0902 and established as an independent project.

### Changed
- Rebranded from "MLX DeepSeek-OCR" to "OCR Butterfly"
- Translated all documentation from Traditional Chinese to English
- Made OCR prompts language-neutral (removed hardcoded Traditional Chinese)
- Updated all SPDX license headers to reference OCR Butterfly
- Rewrote README.md as professional English documentation

### Added
- `ATTRIBUTION.md` — explicit upstream credit
- `THIRD_PARTY_NOTICES.md` — dependency license audit
- `CHANGELOG.md` — this file
- `.editorconfig` — consistent editor formatting
- `pyproject.toml` — linter and formatter configuration
- Improved `.gitignore` (IDE files, build artifacts, `.app` bundles)

### Removed
- `build_app.sh` — packaging script (not relevant for source distribution)
- `docs/` — Chinese-only documentation files (replaced by English README)
- Hardcoded Traditional Chinese in OCR prompts
