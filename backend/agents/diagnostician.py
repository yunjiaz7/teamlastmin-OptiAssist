from __future__ import annotations

"""
diagnostician.py

Runs MedGemma inference to produce a structured medical diagnosis for a retinal
image. Accepts raw image bytes and a clinical question, and returns a parsed JSON
dict containing condition, severity, findings, and recommendations.
"""

import asyncio
import io
import json
import logging
from pathlib import Path

from PIL import Image
from transformers import pipeline

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model source configuration
#
# Option A — Use your own local fine-tuned model:
#   Set USE_LOCAL_MODEL = True and place the model under:
#   backend/models/medgemma-finetuned/
#
# Option B — Use the official HuggingFace model (auto-downloaded):
#   Set USE_LOCAL_MODEL = False
#   Model ID: google/medgemma-4b-it
#   Note: requires HuggingFace login and license agreement.
# ---------------------------------------------------------------------------
USE_LOCAL_MODEL = False

_LOCAL_MODEL_PATH = Path(__file__).parent.parent / "models" / "medgemma-finetuned"
_HF_MODEL_ID = "google/medgemma-4b-it"

if USE_LOCAL_MODEL:
    if not _LOCAL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Local MedGemma model not found at '{_LOCAL_MODEL_PATH}'. "
            "Please place the fine-tuned model files in that directory, "
            "or set USE_LOCAL_MODEL = False to use the HuggingFace model."
        )
    MODEL_PATH = str(_LOCAL_MODEL_PATH)
    logger.info("Using local MedGemma model from %s", MODEL_PATH)
else:
    MODEL_PATH = _HF_MODEL_ID
    logger.info("Using HuggingFace MedGemma model: %s", MODEL_PATH)

SYSTEM_PROMPT = (
    "You are an expert ophthalmology AI assistant. "
    "Analyze the retinal image and answer the clinical question.\n\n"
    "CRITICAL INSTRUCTION: Your response must be ONLY a valid JSON object. "
    "Do NOT include any explanation, preamble, markdown, or text outside the JSON. "
    "Your response must start with { and end with }.\n\n"
    "Required JSON fields:\n"
    "  condition: string (disease name or 'Normal')\n"
    "  severity: string ('None', 'Mild', 'Moderate', 'Severe', or 'Proliferative')\n"
    "  severity_level: integer 0-4\n"
    "  confidence: float 0.0-1.0\n"
    "  findings: list of strings (specific observations)\n"
    "  recommendation: string (follow-up advice)\n"
    "  disclaimer: string (always 'For research use only. Not intended for clinical diagnosis.')\n\n"
    "Example response:\n"
    '{"condition": "Diabetic Retinopathy", "severity": "Moderate", "severity_level": 2, '
    '"confidence": 0.85, "findings": ["microaneurysms", "hard exudates", "cotton wool spots"], '
    '"recommendation": "Refer to retinal specialist within 1 month.", '
    '"disclaimer": "For research use only. Not intended for clinical diagnosis."}'
)

FALLBACK_RESULT = {
    "condition": "Analysis unavailable",
    "severity": "None",
    "severity_level": 0,
    "confidence": 0.0,
    "findings": [],
    "recommendation": "Please retry or consult a qualified ophthalmologist.",
    "disclaimer": "For research use only. Not intended for clinical diagnosis.",
}

# ---------------------------------------------------------------------------
# Model loading — happens once at module level to avoid per-call overhead
# ---------------------------------------------------------------------------
logger.info("Loading MedGemma pipeline from %s", MODEL_PATH)
pipe = pipeline("image-text-to-text", model=MODEL_PATH)
logger.info("MedGemma pipeline loaded successfully.")


def _run_inference(messages: list[dict], continue_final_message: bool = False) -> str:
    """
    Execute blocking MedGemma pipeline inference synchronously.

    Args:
        messages: Chat-formatted message list. All content fields must be
            list-of-dicts (never a plain string) so that Gemma3Processor's
            apply_chat_template can iterate over them without hitting
            "string indices must be integers".
        continue_final_message: When True the pipeline continues the last
            assistant message instead of starting a new turn. Used for the
            JSON prefill trick: the caller appends an assistant message whose
            content starts with "{", ensuring the model cannot output any
            preamble text before the JSON.

    Returns:
        Raw text content of the model's last response turn.
    """
    output = pipe(
        text=messages,
        max_new_tokens=512,
        continue_final_message=continue_final_message,
    )
    generated = output[0]["generated_text"]

    if isinstance(generated, list):
        # Standard chat format: list of {"role": ..., "content": ...} dicts.
        # The last element is the model's assistant turn.
        last = generated[-1]
        if isinstance(last, dict):
            content = last.get("content")
            if isinstance(content, list):
                # With continue_final_message=True the pipeline returns content
                # in list-of-dicts format: [{"type": "text", "text": "..."}].
                # Extract and concatenate all text parts.
                return "".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ) or str(last)
            # Plain string content — return directly.
            return str(content) if content else str(last)
        # Fallback: list of plain strings — take the last one.
        return str(last)

    # Plain string — the pipeline returned the completion text directly.
    return str(generated)


def _parse_json(raw_text: str) -> dict:
    """
    Parse a JSON dict from the model's raw output string.

    Tries strict json.loads first; if that fails, falls back to extracting
    the substring between the first '{' and last '}' before retrying.
    Returns FALLBACK_RESULT if both attempts fail.

    Args:
        raw_text: The raw string output from the MedGemma pipeline.

    Returns:
        Parsed diagnosis dict, or FALLBACK_RESULT on failure.
    """
    # Attempt 1: direct parse (model followed instructions and returned clean JSON)
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        pass

    # Attempt 2: extract the outermost JSON object from the string
    start = raw_text.find("{")
    end = raw_text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw_text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("JSON parsing failed for model output. Returning fallback result.")
    return FALLBACK_RESULT


async def run_diagnosis(image_bytes: bytes | None, query: str) -> dict:
    """
    Produce a structured ophthalmological diagnosis using MedGemma.

    Args:
        image_bytes: Raw bytes of the retinal image, or None for text-only queries.
        query: The clinician's diagnostic question.

    Returns:
        A dict with keys: condition, severity, severity_level, confidence,
        findings, recommendation, disclaimer.

    Raises:
        RuntimeError: If inference fails unexpectedly.
    """
    pil_image: Image.Image | None = None
    if image_bytes is not None:
        try:
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to decode image bytes into PIL Image: {e}") from e

    # Build the user message, attaching the image when available.
    # IMPORTANT: all content fields MUST be list-of-dicts (never a plain string).
    # The Gemma3Processor.apply_chat_template does:
    #   visuals = [c for c in message["content"] if c["type"] in ["image","video"]]
    # If content is a plain string, iterating over it yields characters, and
    # character["type"] raises "string indices must be integers".
    user_content: list[dict]
    if pil_image is not None:
        user_content = [
            {"type": "image", "image": pil_image},
            {"type": "text", "text": query},
        ]
    else:
        user_content = [{"type": "text", "text": query}]

    # Prefill the assistant turn with "{" so the model is physically forced to
    # continue with a JSON object and cannot output any preamble text.
    # continue_final_message=True tells the pipeline not to close this turn,
    # allowing the model to generate the rest of the JSON from there.
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": [{"type": "text", "text": "{"}]},
    ]

    logger.info("Running MedGemma diagnosis. query=%s has_image=%s", query[:80], pil_image is not None)

    try:
        # Offload blocking pipeline call to a thread to keep the event loop free
        raw_text = await asyncio.to_thread(_run_inference, messages, True)
    except Exception as e:
        raise RuntimeError(f"MedGemma inference failed: {e}") from e

    logger.debug("Raw MedGemma output: %s", raw_text[:200])

    result = _parse_json(raw_text)
    logger.info("Diagnosis complete. condition=%s severity=%s", result.get("condition"), result.get("severity"))

    return result
