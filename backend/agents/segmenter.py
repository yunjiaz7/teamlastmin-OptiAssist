from __future__ import annotations

"""
segmenter.py

PaliGemma 2 agent — detects optic disc and optic cup bounding boxes from a
retinal fundus image.

This agent uses a fine-tuned PaliGemma 2 (3B, LoRA-adapted) model that was
specifically trained on the optic-disc / optic-cup detection task.  Given a
fundus image it returns pixel-space bounding boxes for both structures, which
are subsequently consumed by the cup-to-disc ratio metric tools.

Model details
-------------
  Base model:  google/paligemma2-3b-pt-448
  Adapter:     finetuned_paligemma2_det_lora  (LoRA, merged at inference time)
  Task prompt: "detect optic-disc ; optic-cup"
  Output:      <loc####> token sequences → parsed into {x_min, y_min, x_max, y_max}

Agent role in the pipeline
---------------------------
  1. MedGemma runs a general medical diagnosis on the fundus image.
  2. If the diagnosis involves the optic disc (e.g. glaucoma, cupping),
     THIS agent is called to precisely locate the disc and cup.
  3. The returned bounding boxes are fed into the CDR metric tools
     (compute_vertical_cdr, compute_horizontal_cdr, compute_area_cdr, etc.).
"""

import asyncio
import io
import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# LoRA adapter path
# Fine-tuned specifically for: "detect optic-disc ; optic-cup"
# ---------------------------------------------------------------------------
_ADAPTER_DIR = (
    Path(__file__).parent.parent
    / "models"
    / "paligemma2-finetuned"
    / "finetuned_paligemma2_det_lora"
    / "final"
)

# Make sure the project root is on sys.path so `app.tools.paligemma_tool`
# can be imported from inside the backend package.
_PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# The detection prompt this LoRA was trained on — do not change.
_DETECTION_PROMPT = "detect optic-disc ; optic-cup"


def _run_inference_sync(image_bytes: bytes) -> dict:
    """
    Blocking wrapper — loads the model once (cached inside paligemma_tool)
    then runs optic disc and optic cup detection.

    Called via ``asyncio.to_thread`` so the event loop stays free.

    Args:
        image_bytes: Raw bytes of the input fundus image.

    Returns:
        Raw result dict from run_paligemma_detection.
    """
    import os
    import tempfile

    from app.tools.paligemma_tool import run_paligemma_detection

    # paligemma_tool expects a file path — write bytes to a temp file.
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = run_paligemma_detection(
            image_path=tmp_path,
            query_context=_DETECTION_PROMPT,
            max_new_tokens=128,
            adapter_dir=_ADAPTER_DIR,
        )
    finally:
        os.unlink(tmp_path)

    return result


async def run_segmentation(image_bytes: bytes, query: str = _DETECTION_PROMPT) -> dict:
    """
    Detect optic disc and optic cup bounding boxes in a retinal fundus image
    using the fine-tuned PaliGemma 2 model.

    The ``query`` parameter is accepted for API compatibility but is always
    overridden with the fixed detection prompt the model was fine-tuned on
    (``"detect optic-disc ; optic-cup"``).  Do not pass arbitrary lesion names —
    this model was not trained for general-purpose lesion detection.

    Args:
        image_bytes: Raw bytes of the input retinal fundus image (JPEG/PNG).
        query:       Ignored — the model always uses its fine-tuned prompt.

    Returns:
        A dict with keys:
            ``"detections"``            (list[dict]): Each entry has
                ``"label"`` (``"optic-disc"`` or ``"optic-cup"``) and
                ``"bounding_box"`` (``{x_min, y_min, x_max, y_max}`` in pixels).
            ``"annotated_image_base64"`` (str):  Base64 PNG with boxes drawn.
            ``"raw_output"``            (str):  Raw <loc####> token string.
            ``"summary"``               (str):  Human-readable detection summary.

    Raises:
        RuntimeError: If the image cannot be decoded or PaliGemma inference fails.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        w, h = pil.size
    except Exception as e:
        raise RuntimeError(f"Failed to decode image: {e}") from e

    logger.info(
        "PaliGemma 2 — detecting optic disc and cup. image=%dx%d prompt=%r",
        w, h, _DETECTION_PROMPT,
    )

    try:
        result = await asyncio.to_thread(_run_inference_sync, image_bytes)
    except Exception as e:
        raise RuntimeError(f"PaliGemma 2 inference failed: {e}") from e

    detections = result.get("detections", [])
    labels_found = [d.get("label", "") for d in detections]
    logger.info(
        "PaliGemma 2 done. detections=%d labels=%s",
        len(detections), labels_found,
    )
    return result
