# Lessons

- When forcing local-only PaliGemma inference, keep adapter weights local but load the processor/tokenizer from the base model ID (`google/paligemma2-3b-pt-448`) with `local_files_only=True`; tokenizer artifacts saved inside LoRA `final/` can be incompatible with runtime `transformers` and crash (`extra_special_tokens` list vs dict expectations).
- Keep inference base-model load settings aligned with `backend/scripts/train_paligemma.py` unless explicitly testing alternatives: use `PaliGemmaProcessor.from_pretrained(..., use_fast=True)` and `PaliGemmaForConditionalGeneration.from_pretrained(..., device_map/torch_dtype pattern)` to avoid train/infer drift.
