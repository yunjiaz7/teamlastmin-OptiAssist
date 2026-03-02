# TODO

- [x] Confirm PaliGemma train/infer format mismatch root cause
- [x] Patch shared PaliGemma inference tool for base model + prompt/decoding consistency
- [x] Run lints on edited files and summarize verification/results

# Review

- Root-cause checks: training script uses `<image> ...` prompt and parses `<loc...>` by semicolon-delimited detections; inference tool previously diverged and could silently swap to an incompatible 224 base.
- Fixes applied in `app/tools/paligemma_tool.py`:
  - Removed silent fallback from `google/paligemma2-3b-pt-448` to `...pt-224`.
  - Aligned inference prompt to training format (`<image> detect ...`).
  - Aligned processor call with `truncation=False` and generated-token suffix decoding.
  - Kept coordinate parsing consistent with train-time detection-string format.
- Verification:
  - `read_lints` on edited file: no lints.
  - `python3 -m py_compile app/tools/paligemma_tool.py` succeeded with workspace-local `PYTHONPYCACHEPREFIX`.

---

## Frontend PaliGemma result image

- [x] Confirm backend event/result payload already includes `annotated_image_base64` and identify the safest field path for frontend consumption.
- [x] Add segmentation image field(s) to frontend types + SSE parsing for `paligemma_complete`.
- [x] Render the PaliGemma result image card in the demo pipeline UI with graceful fallback when absent.
- [x] Run lints on edited frontend/backend files and record verification notes.

## Review

- Backend update in `backend/orchestrator.py`: `paligemma_complete` event payload now forwards `annotated_image_base64` from segmentation output.
- Frontend update in `frontend/app/demo/page.tsx`: `SegmentationData` now includes `annotated_image_base64` and `PaliGemmaSegmentationCard` renders that image in the pipeline panel.
- Existing parsing path (`segmentationData = parsed.segmentation`) automatically picks up the new field, so no extra transform logic was needed.
- Verification: `read_lints` run on edited files (`frontend/app/demo/page.tsx`, `backend/orchestrator.py`, `tasks/todo.md`) reports no linter errors.
