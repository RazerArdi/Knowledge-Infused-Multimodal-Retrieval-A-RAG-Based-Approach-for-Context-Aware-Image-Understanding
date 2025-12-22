# Future Roadmap & TODOs

## Immediate Priorities
- [ ] Add support for Audio retrieval (using AudioCLIP).
- [ ] Optimize FAISS index for >1 Million images (IVF_PQ).

## Long-term Goals
- [ ] Deploy as a Docker container for easier distribution.
- [ ] Implement "Re-Ranking" mechanism after retrieval to improve accuracy.
- [ ] Upgrade BLIP-2 to LLaVA or GPT-4o API integration.

## Known Limitations
- Current inference speed is ~70s on RTX 3060 (Need optimization via TensorRT).
- LoRA Fine-tuning limited to domain-specific captions only.