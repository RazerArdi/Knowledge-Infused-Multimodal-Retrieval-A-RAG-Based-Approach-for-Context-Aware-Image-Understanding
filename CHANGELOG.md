# Changelog

All notable changes to the "Multimodal RAG System" project will be documented in this file.

## [Unreleased]
- Integration with Llama-3 70B for better reasoning.
- Support for PDF document retrieval (Multimodal + Text).

## [1.0.0] - 2024-12-22
### Added
- **Fine-Tuning Pipeline:** Added QLoRA support for BLIP-2 (`Notebook/FinalProject_Multimodal_RAG.ipynb`).
- **Academic Metrics:** Implemented BLEU-4 and ROUGE-L for caption evaluation.
- **RAG Metrics:** Implemented Faithfulness and Answer Relevance using CLIP Embeddings.
- **Auto-Loader:** `app.py` now automatically detects and merges LoRA adapters.

### Changed
- Switched base retrieval model from `clip-vit-base` to `clip-vit-large` for better accuracy.
- Optimized memory usage for 8GB VRAM GPU (Batch size=1, Grad Accumulation=8).

### Fixed
- Fixed `peft` config conflict with multimodal inputs.
- Resolved OOM (Out of Memory) issues during generation.