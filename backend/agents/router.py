"""
router.py

Routes an incoming user request to the correct analysis function by calling
FunctionGemma via the Ollama local API. The model selects from three tools
(analyze_location, analyze_diagnosis, analyze_full) based on the user's question
and the pre-scanned image description.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "functiongemma"
DEFAULT_ROUTE = "analyze_full"

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "analyze_location",
            "description": (
                "Use when user wants to locate, detect, or segment a specific anatomical "
                "structure or lesion in the retinal image. "
                "Trigger keywords: where, locate, show me, detect, segment, find"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the location query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_diagnosis",
            "description": (
                "Use when user wants a medical judgment, classification, disease identification, "
                "or risk assessment. "
                "Trigger keywords: is this, diagnosis, what disease, severity, risk, normal, condition"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the diagnostic query",
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_full",
            "description": (
                "Use when user wants both location information AND medical diagnosis together. "
                "Trigger keywords: full analysis, everything, what is wrong and where, complete report"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the full analysis query",
                    }
                },
                "required": ["query"],
            },
        },
    },
]


async def route_request(question: str, image_description: str = "") -> dict:
    """
    Determine which analysis function best matches the user's request.

    Args:
        question: The clinician's question string.
        image_description: Optional pre-scan description of the image content.

    Returns:
        A dict with keys:
            "function" (str): One of "analyze_location", "analyze_diagnosis", "analyze_full".
            "query"    (str): The query argument extracted from the tool call.
        Falls back to {"function": "analyze_full", "query": question} if routing fails.
    """
    # Combine question and image context so the model has full information to decide
    combined_context = question
    if image_description:
        combined_context = f"{question}\n\nImage context: {image_description}"

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [{"role": "user", "content": combined_context}],
        "tools": TOOLS,
        "stream": False,
    }

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OLLAMA_URL, json=payload)
            response.raise_for_status()
            data = response.json()

        tool_calls = (
            data.get("message", {}).get("tool_calls") or []
        )

        if not tool_calls:
            raise ValueError("No tool call returned by model.")

        first_call = tool_calls[0]
        tool_name = first_call["function"]["name"]
        arguments = first_call["function"].get("arguments", {})
        query = arguments.get("query", question)

        logger.info("Router selected function=%s query=%s", tool_name, query[:80])
        return {"function": tool_name, "query": query}

    except Exception as e:
        # Default to full analysis so the user always gets a useful response
        logger.warning("Routing failed, defaulting to analyze_full. reason=%s", str(e))
        return {"function": DEFAULT_ROUTE, "query": question}
