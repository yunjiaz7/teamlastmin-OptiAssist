from __future__ import annotations

"""
segmenter.py

Runs PaliGemma 2 inference for retinal image segmentation.
Delegates all model loading and inference to app.tools.paligemma_tool
so the logic stays in one place and run_single_inference.py stays usable.
"""

import asyncio
import io
import logging
import sys
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)

# Path to the LoRA adapter's final/ directory
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


def _run_inference_sync(image_bytes: bytes, query: str) -> dict:
    """
    Blocking wrapper — loads the model once (cached inside paligemma_tool)
    then runs inference.  Called via asyncio.to_thread so the event loop
    stays free.
    """
    import tempfile, os
    from app.tools.paligemma_tool import run_paligemma_detection

    # paligemma_tool expects a file path, so write bytes to a temp file
    suffix = ".jpg"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(image_bytes)
        tmp_path = tmp.name

    try:
        result = run_paligemma_detection(
            image_path=tmp_path,
            query_context=query,
            max_new_tokens=128,
            adapter_dir=_ADAPTER_DIR,
        )
    finally:
        os.unlink(tmp_path)

    return result


async def run_segmentation(image_bytes: bytes, query: str) -> dict:
    """
    Segment anatomical structures in a retinal image using PaliGemma 2.

    Args:
        image_bytes: Raw bytes of the input retinal image.
        query:       Detection query (e.g. "optic disc", "microaneurysms").

    Returns:
        A dict with keys:
            "detections"             (list): Detected regions with bounding boxes.
            "annotated_image_base64" (str):  Base64 PNG with boxes drawn.
            "summary"                (str):  Human-readable detection summary.
    """
    try:
        pil = Image.open(io.BytesIO(image_bytes))
        w, h = pil.size
    except Exception as e:
        raise RuntimeError(f"Failed to decode image: {e}") from e

    logger.info("Running segmentation. query=%r image=%dx%d", query, w, h)

    try:
        result = await asyncio.to_thread(_run_inference_sync, image_bytes, query)
    except Exception as e:
        raise RuntimeError(f"PaliGemma 2 inference failed: {e}") from e

    logger.info("Segmentation done. detections=%d", len(result.get("detections", [])))
    return result
