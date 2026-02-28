"""
prescanner.py

Pre-scans a retinal image using Gemma 3 via the Ollama local API and returns
a brief natural-language description of the image content. This description is
passed downstream to the router to help select the appropriate analysis path.
"""

import base64
import logging

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
PRESCAN_PROMPT = (
    "Describe this medical retinal image in 1-2 sentences. "
    "Focus on visible structures and any abnormalities. Be factual and concise."
)
FALLBACK_DESCRIPTION = "Retinal fundus image"


async def prescan_image(image_bytes: bytes) -> str:
    """
    Send a retinal image to Gemma 3 (via Ollama) and return a short description.

    Args:
        image_bytes: Raw bytes of the input retinal image.

    Returns:
        A 1-2 sentence plain-text description of the image.
        Falls back to a generic label if the Ollama call fails.
    """
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": PRESCAN_PROMPT,
        "images": [base64_image],
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()
            description = data["response"]
            logger.info("Prescan succeeded. description=%s", description[:80])
            return description
    except Exception as e:
        # Fall back gracefully so the pipeline can continue with a safe default
        logger.warning("Prescan failed, using fallback description. reason=%s", str(e))
        return FALLBACK_DESCRIPTION
