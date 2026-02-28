from __future__ import annotations

"""
segmenter.py

Runs PaliGemma 2 inference for retinal image segmentation. Accepts raw image
bytes and a location query, returns detected anatomical regions with bounding
boxes drawn on an annotated copy of the original image.

Mask decoding (from <seg> tokens) requires vae-oid.npz — coordinate with teammate.
Until that asset is available, has_mask is always set to False.
"""

import asyncio
import base64
import io
import logging
import os
import re
import traceback
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image, ImageDraw
from transformers import AutoModelForCausalLM, AutoProcessor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model source configuration
#
# Option A — Use the local LoRA fine-tuned adapter (default):
#   Set USE_LOCAL_MODEL = True. The adapter lives at:
#   backend/models/paligemma2-finetuned/finetuned_paligemma2_det_lora/final/
#   The base model (google/paligemma2-3b-pt-448) is downloaded from HuggingFace
#   and the LoRA weights are applied on top.
#
# Option B — Use the bare HuggingFace base model (no adapter):
#   Set USE_LOCAL_MODEL = False
#   Model ID: google/paligemma2-3b-pt-448
#   Note: requires HuggingFace login and license agreement.
# ---------------------------------------------------------------------------
USE_LOCAL_MODEL = True

_LOCAL_ADAPTER_PATH = (
    Path(__file__).parent.parent
    / "models"
    / "paligemma2-finetuned"
    / "finetuned_paligemma2_det_lora"
    / "final"
)

# The LoRA adapter was trained on pt-448, but only pt-224 is cached locally.
# The LoRA targets only language-model layers (q/k/v/o/gate/up/down proj),
# whose weight shapes are identical between 224 and 448.  We therefore fall
# back to the cached pt-224 base so we can run offline.
_HF_MODEL_448 = "google/paligemma2-3b-pt-448"
_HF_MODEL_224 = "google/paligemma2-3b-pt-224"

def _pick_base_model() -> str:
    """Return the HF model ID to use as the LoRA base.

    Prefers pt-448 (matches fine-tune resolution) but falls back to the
    locally-cached pt-224 so the server can start without a network download.
    """
    import huggingface_hub
    cache_info = huggingface_hub.scan_cache_dir()
    cached_ids = {repo.repo_id for repo in cache_info.repos}
    if _HF_MODEL_448 in cached_ids:
        logger.info("Found cached pt-448 model, using it as LoRA base.")
        return _HF_MODEL_448
    if _HF_MODEL_224 in cached_ids:
        logger.warning(
            "pt-448 not in HF cache — falling back to pt-224 as LoRA base. "
            "Image resolution will be 224 px instead of 448 px."
        )
        return _HF_MODEL_224
    # Neither cached: attempt pt-448 download (requires HF auth + network)
    logger.warning("Neither pt-448 nor pt-224 found in HF cache. Attempting download of pt-448.")
    return _HF_MODEL_448

if USE_LOCAL_MODEL:
    if not _LOCAL_ADAPTER_PATH.exists():
        raise FileNotFoundError(
            f"Fine-tuned LoRA adapter not found at '{_LOCAL_ADAPTER_PATH}'. "
            "Please unzip finetuned_paligemma2_det_lora.zip into "
            "backend/models/paligemma2-finetuned/, "
            "or set USE_LOCAL_MODEL = False to use the bare HuggingFace model."
        )
    ADAPTER_PATH = str(_LOCAL_ADAPTER_PATH)
    _HF_BASE_MODEL_ID = _pick_base_model()
    logger.info("Using local LoRA adapter from %s on top of base model %s", ADAPTER_PATH, _HF_BASE_MODEL_ID)
else:
    ADAPTER_PATH = None
    _HF_BASE_MODEL_ID = _HF_MODEL_448
    logger.info("Using HuggingFace PaliGemma 2 base model: %s", _HF_BASE_MODEL_ID)

# ---------------------------------------------------------------------------
# Model loading — happens once at module level to avoid per-call overhead.
#
# For the LoRA path:
#   1. Load the processor from the base model ID (guaranteed correct config).
#   2. Load the base model weights.
#   3. Apply the LoRA adapter with PeftModel.from_pretrained().
#   4. merge_and_unload() folds the LoRA delta weights into the base weights.
# ---------------------------------------------------------------------------
try:
    logger.info("Loading processor from base model %s", _HF_BASE_MODEL_ID)
    processor = AutoProcessor.from_pretrained(_HF_BASE_MODEL_ID)

    if USE_LOCAL_MODEL:
        logger.info("Loading base model weights from %s", _HF_BASE_MODEL_ID)
        _base_model = AutoModelForCausalLM.from_pretrained(
            _HF_BASE_MODEL_ID, torch_dtype=torch.float32
        )
        logger.info("Applying LoRA adapter from %s ...", ADAPTER_PATH)
        model = PeftModel.from_pretrained(_base_model, ADAPTER_PATH)
        model = model.merge_and_unload()
        model.eval()
        logger.info("LoRA adapter merged. PaliGemma 2 fine-tuned model ready.")
    else:
        logger.info("Loading base model weights from %s", _HF_BASE_MODEL_ID)
        model = AutoModelForCausalLM.from_pretrained(
            _HF_BASE_MODEL_ID, torch_dtype=torch.float32
        )
        model.eval()
        logger.info("PaliGemma 2 base model loaded successfully.")
except Exception:
    logger.error("Model loading FAILED:\n%s", traceback.format_exc())
    raise

# Regex to capture all <loc####> tokens in sequence
_LOC_PATTERN = re.compile(r"<loc(\d{4})>")


def _run_inference(pil_image: Image.Image, prompt: str) -> str:
    """
    Execute blocking PaliGemma 2 inference synchronously.

    Args:
        pil_image: PIL Image object of the retinal scan.
        prompt: Text prompt for the model (e.g. "detect optic disc\\n").

    Returns:
        Raw decoded output string (generated tokens only, special tokens kept).
    """
    inputs = processor(text=prompt, images=pil_image, return_tensors="pt")
    input_len = inputs["input_ids"].shape[-1]
    logger.info(
        "Processor output keys: %s  input_ids shape: %s",
        list(inputs.keys()),
        inputs["input_ids"].shape,
    )

    with torch.inference_mode():
        # use_cache=False avoids a past_key_values list-vs-dict mismatch that
        # appears in some transformers versions with PaliGemma 2.
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_cache=False,
        )

    # Decode only the newly generated tokens (strip the echoed input prompt)
    generated_tokens = outputs[0][input_len:]
    raw_output = processor.decode(generated_tokens, skip_special_tokens=False)
    logger.info("Raw PaliGemma output: %s", raw_output[:300])
    return raw_output


def _parse_detections(raw_output: str, img_width: int, img_height: int) -> list[dict]:
    """
    Parse <loc####> tokens from raw PaliGemma output into structured detections.

    PaliGemma encodes bounding boxes as four consecutive <loc####> tokens in the
    order: y_min, x_min, y_max, x_max. Each token value is in [0, 1023] and must
    be scaled to the actual image dimensions.

    Args:
        raw_output: Full decoded model output string with special tokens intact.
        img_width:  Width of the original image in pixels.
        img_height: Height of the original image in pixels.

    Returns:
        List of detection dicts, each containing label, bounding_box, and has_mask.
    """
    loc_tokens = _LOC_PATTERN.findall(raw_output)
    detections = []

    # Group loc tokens in blocks of 4; anything left over is incomplete and skipped
    for i in range(0, len(loc_tokens) - 3, 4):
        y_min_raw, x_min_raw, y_max_raw, x_max_raw = (int(loc_tokens[i + j]) for j in range(4))

        y_min = int((y_min_raw / 1024) * img_height)
        x_min = int((x_min_raw / 1024) * img_width)
        y_max = int((y_max_raw / 1024) * img_height)
        x_max = int((x_max_raw / 1024) * img_width)

        # Attempt to extract a label string following the 4th loc token in the raw text
        after_locs = re.split(r"(?:<loc\d{4}>){4}", raw_output)
        label_text = ""
        label_idx = i // 4 + 1
        if label_idx < len(after_locs):
            # Take only the first word/phrase before the next special token
            label_candidate = re.split(r"<", after_locs[label_idx])[0].strip()
            label_text = label_candidate if label_candidate else f"region_{label_idx}"

        # <seg> tokens indicate mask availability; decoding requires vae-oid.npz (TBD)
        has_mask = False

        detections.append(
            {
                "label": label_text or f"region_{i // 4 + 1}",
                "confidence": 0.9,
                "bounding_box": {
                    "x_min": x_min,
                    "y_min": y_min,
                    "x_max": x_max,
                    "y_max": y_max,
                },
                "has_mask": has_mask,
            }
        )

    return detections


def _draw_boxes(pil_image: Image.Image, detections: list[dict]) -> str:
    """
    Draw red bounding boxes on the image and encode it as a base64 PNG data URI.

    Args:
        pil_image:  Original PIL Image to annotate.
        detections: List of detection dicts produced by _parse_detections().

    Returns:
        Base64-encoded PNG string prefixed with the data URI scheme.
    """
    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)

    for det in detections:
        box = det["bounding_box"]
        draw.rectangle(
            [box["x_min"], box["y_min"], box["x_max"], box["y_max"]],
            outline="red",
            width=2,
        )

    buffer = io.BytesIO()
    annotated.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_summary(detections: list[dict]) -> str:
    """
    Produce a human-readable summary of detected regions.

    Args:
        detections: List of detection dicts.

    Returns:
        A plain-text summary string.
    """
    if not detections:
        return "No regions detected."

    labels = [d["label"] for d in detections]
    label_phrases = ", ".join(f"{lbl} in region" for lbl in labels)
    return f"{len(detections)} regions detected: {label_phrases}"


async def run_segmentation(image_bytes: bytes, query: str) -> dict:
    """
    Segment anatomical structures in a retinal image using PaliGemma 2.

    Args:
        image_bytes: Raw bytes of the input retinal image.
        query: Location query describing what to segment (e.g. "optic disc").

    Returns:
        A dict with keys:
            "detections"            (list): Detected regions with labels and bounding boxes.
            "annotated_image_base64" (str): Base64 PNG of the image with boxes drawn.
            "summary"               (str): Human-readable detection summary.

    Raises:
        RuntimeError: If inference or post-processing fails unexpectedly.
    """
    try:
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise RuntimeError(f"Failed to decode image bytes into PIL Image: {e}") from e

    img_width, img_height = pil_image.size
    # Fine-tune was trained as a detection task — use "detect" prefix
    prompt = f"detect {query}\n"

    logger.info("Running segmentation. query=%s image_size=%dx%d", query, img_width, img_height)

    try:
        # Offload blocking transformer inference to a thread so the event loop stays free
        raw_output = await asyncio.to_thread(_run_inference, pil_image, prompt)
    except Exception as e:
        logger.error("PaliGemma 2 inference FULL traceback:\n%s", traceback.format_exc())
        raise RuntimeError(f"PaliGemma 2 inference failed: {e}") from e

    logger.debug("Raw model output: %s", raw_output[:200])

    detections = _parse_detections(raw_output, img_width, img_height)
    annotated_image_base64 = _draw_boxes(pil_image, detections)
    summary = _build_summary(detections)

    logger.info("Segmentation complete. detections=%d", len(detections))

    return {
        "detections": detections,
        "annotated_image_base64": annotated_image_base64,
        "summary": summary,
    }
