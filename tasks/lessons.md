# Lessons Learned

- When a user pivots scope (for example from notebook edits to standalone scripts), update plans/status files in the same pass so documentation stays aligned with implementation.
- Prefer creating small reusable entrypoints (like dataset-only conversion scripts) instead of leaving placeholder files in the repo.
- If users request research-only framing, ensure all prompts and returned disclaimers consistently avoid clinical-use wording and align final synthesis model choice with the request.
- For demo/POC workflows, avoid returning extra disclaimer fields or prompt-forced caution text unless explicitly requested by the user.
- When model capabilities differ (for example image vs text-only), encode explicit routing overrides and fallback model chains to avoid runtime tool-selection failures.
- When the user asks to remove a model fallback (for example medgemma3), align all planner/synthesis code paths and prompts to a single intended model and remove obsolete route-specific heuristics.
- For Ollama vision models, prefer passing image paths (or raw bytes) in `messages[].images` instead of base64-only payloads to avoid model template image-input errors.
