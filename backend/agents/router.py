from __future__ import annotations

"""
router.py

Implements a multi-turn FunctionGemma agentic loop for OptiAssist.

FunctionGemma maintains a messages conversation history and autonomously decides
which tools to call and in what order, until it either returns a plain text
response or explicitly calls the 'finish' tool.

Message format follows the official FunctionGemma spec:
  - System prompt uses role "developer"
  - Tool results use role "tool" with content [{"name": ..., "response": ...}]
  - After a tool call the assistant message (with tool_calls) is appended before
    the tool result, preserving the correct alternating turn structure.

Reference: https://ai.google.dev/gemma/docs/functiongemma/full-function-calling-sequence-with-functiongemma
"""

import json
import logging
from typing import Awaitable, Callable

import httpx

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "functiongemma"

MAX_LOOP_ITERATIONS = 5

# Maximum number of times we nudge FunctionGemma back onto the tool-calling
# path when it tries to answer directly instead of calling a tool.
MAX_NUDGE_RETRIES = 2

# ---------------------------------------------------------------------------
# System prompt
#
# IMPORTANT: FunctionGemma was fine-tuned on this EXACT developer message.
# It acts as a mode-activation trigger. Replacing it with a different string
# prevents the model from entering function-calling mode and causes it to
# answer in plain text instead. Keep this verbatim.
# Reference: https://ai.google.dev/gemma/docs/functiongemma/
#            full-function-calling-sequence-with-functiongemma
# ---------------------------------------------------------------------------
DEVELOPER_MESSAGE = (
    "You are a model that can do function calling with the following functions"
)

# Orchestration instructions live in the USER message, not the developer
# message, so they don't interfere with the function-calling activation signal.
_ORCHESTRATION_INSTRUCTIONS = (
    "\n\n---\n"
    "INSTRUCTIONS — you MUST follow these exactly:\n"
    "1. You are NOT allowed to answer the question directly.\n"
    "2. You MUST call tools to complete the analysis.\n"
    "3. Your FIRST action MUST be to call run_diagnosis.\n"
    "4. After run_diagnosis returns:\n"
    "   a. If findings contain abnormal lesions → call run_segmentation "
    "with the SPECIFIC lesion name from the findings "
    "(e.g. 'cotton wool spots', 'microaneurysms'). "
    "Do NOT pass the user's original question as the query.\n"
    "   b. If the condition is Normal or findings are empty → call finish.\n"
    "5. After all needed tools have run → call finish.\n"
    "Begin now by calling run_diagnosis."
)

# Message injected when FunctionGemma answers in text instead of calling a tool
_NUDGE_MESSAGE = (
    "You responded with text instead of calling a tool. "
    "This is not allowed. "
    "You MUST call run_diagnosis right now. "
    "Do not output any text — only call the run_diagnosis tool."
)

# ---------------------------------------------------------------------------
# Tool definitions exposed to FunctionGemma
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_diagnosis",
            "description": (
                "Run MedGemma 4B to analyze the retinal image and produce a "
                "structured medical diagnosis including condition, severity, "
                "findings, and recommendations. Always call this first."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_segmentation",
            "description": (
                "Run PaliGemma 2 to locate and segment specific lesions or "
                "structures in the retinal image. Only call after run_diagnosis "
                "has returned abnormal findings. The query must be a specific "
                "pathology name from the diagnosis (e.g. 'cotton wool spots', "
                "'microaneurysms'), not the user's original question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "Specific lesion or anatomical structure to segment, "
                            "derived from diagnosis findings "
                            "(e.g. 'cotton wool spots', 'optic disc edema')."
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": (
                "Signal that the analysis workflow is complete. Call this after "
                "all necessary agents have been invoked."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Map first tool call → route name understood by the frontend AgentDecisionCard
_TOOL_TO_ROUTE: dict[str, str] = {
    "run_diagnosis": "analyze_diagnosis",
    "run_segmentation": "analyze_location",
    "finish": "analyze_diagnosis",
}


async def _call_functiongemma(messages: list[dict]) -> dict:
    """
    Send the current messages history to FunctionGemma via Ollama.

    Args:
        messages: Full conversation history, correctly alternating user / assistant
                  / tool turns.

    Returns:
        The "message" object from Ollama's response (contains "content" and/or
        "tool_calls").

    Raises:
        httpx.HTTPStatusError: If Ollama returns a non-2xx response.
    """
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "tools": TOOLS,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
    return data.get("message", {})


async def run_agentic_loop(
    question: str,
    image_description: str,
    run_diagnosis_cb: Callable[[], Awaitable[dict]],
    run_segmentation_cb: Callable[[str], Awaitable[dict]],
    emit: Callable[[str, str], Awaitable[None]],
) -> dict:
    """
    Run the FunctionGemma multi-turn agentic loop.

    FunctionGemma autonomously decides which tools to invoke and in what order
    within the same conversation context, until it either returns a plain-text
    answer or calls the 'finish' tool.

    Args:
        question:           The clinician's question.
        image_description:  Pre-scanned image description from the prescanner.
        run_diagnosis_cb:   Async callback that runs MedGemma diagnosis;
                            emits medgemma_start / medgemma_complete internally.
        run_segmentation_cb: Async callback(query: str) that runs PaliGemma 2
                            segmentation; emits paligemma_start / paligemma_complete
                            internally.
        emit:               Async SSE emit callback — called as
                            await emit(event: str, message: str).

    Returns:
        Dict with keys:
            "diagnosis"  (dict | None): MedGemma result, or None if not run.
            "location"   (dict | None): PaliGemma result, or None if not run.
            "final_text" (str):         FunctionGemma's last plain-text response.
    """
    # Build initial conversation:
    #   [developer]  exact fine-tuning activation phrase
    #   [user]       question + image context + hard orchestration instructions
    user_content = f"{question}"
    if image_description:
        user_content = f"{question}\n\nImage context: {image_description}"
    user_content += _ORCHESTRATION_INSTRUCTIONS

    messages: list[dict] = [
        {
            "role": "developer",
            "content": DEVELOPER_MESSAGE,
        },
        {
            "role": "user",
            "content": user_content,
        },
    ]

    diagnosis: dict | None = None
    location: dict | None = None
    final_text: str = ""
    route_decided_emitted = False
    nudge_count = 0  # how many times we've nudged the model back to tools

    for iteration in range(MAX_LOOP_ITERATIONS):
        logger.info(
            "Agentic loop — iteration %d/%d", iteration + 1, MAX_LOOP_ITERATIONS
        )

        try:
            message = await _call_functiongemma(messages)
        except Exception as exc:
            logger.warning(
                "FunctionGemma call failed at iteration %d: %s", iteration + 1, exc
            )
            break

        tool_calls: list[dict] = message.get("tool_calls") or []
        text_content: str = message.get("content") or ""

        # ----------------------------------------------------------------
        # No tool calls → FunctionGemma answered in plain text.
        # If we still have nudge budget AND haven't run diagnosis yet,
        # push a reminder and retry rather than accepting the bare text.
        # ----------------------------------------------------------------
        if not tool_calls:
            if diagnosis is None and nudge_count < MAX_NUDGE_RETRIES:
                nudge_count += 1
                logger.warning(
                    "FunctionGemma returned text without calling a tool "
                    "(iteration %d, nudge %d/%d). Injecting reminder.",
                    iteration + 1,
                    nudge_count,
                    MAX_NUDGE_RETRIES,
                )
                # Append the bare assistant reply so the history is valid,
                # then add a hard user-role nudge to force a tool call.
                if text_content:
                    messages.append({"role": "assistant", "content": text_content})
                messages.append({"role": "user", "content": _NUDGE_MESSAGE})
                continue  # retry this iteration slot

            # Out of nudge budget or diagnosis already ran → accept text exit.
            final_text = text_content
            logger.info(
                "FunctionGemma returned text (no tool calls). "
                "nudge_count=%d diagnosis_done=%s. Exiting loop.",
                nudge_count,
                diagnosis is not None,
            )
            break

        # Append the assistant turn (must come before the tool result turn)
        messages.append(message)

        # ----------------------------------------------------------------
        # Process every tool call in this turn.
        # ----------------------------------------------------------------
        tool_results: list[dict] = []
        should_finish = False

        for call in tool_calls:
            fn_name: str = call["function"]["name"]
            fn_args: dict = call["function"].get("arguments") or {}
            logger.info(
                "FunctionGemma called tool=%s args=%s", fn_name, fn_args
            )

            # Emit route_decided once, based on the first meaningful tool call
            if not route_decided_emitted and fn_name in _TOOL_TO_ROUTE:
                route_name = _TOOL_TO_ROUTE[fn_name]
                await emit("route_decided", f"Route: {route_name}")
                logger.info("Route decided: %s", route_name)
                route_decided_emitted = True

            if fn_name == "finish":
                should_finish = True
                tool_results.append({"name": "finish", "response": "Analysis complete."})

            elif fn_name == "run_diagnosis":
                try:
                    diagnosis = await run_diagnosis_cb()
                    tool_results.append({
                        "name": "run_diagnosis",
                        "response": json.dumps(diagnosis),  # must be string, not dict
                    })
                except Exception as exc:
                    logger.error("run_diagnosis_cb raised: %s", exc)
                    # Use a plain string so FunctionGemma's chat template can
                    # format the tool turn correctly. A nested dict in the error
                    # path causes a 400 on the subsequent Ollama call.
                    tool_results.append({
                        "name": "run_diagnosis",
                        "response": f"Tool execution failed: {exc}",
                    })

            elif fn_name == "run_segmentation":
                query: str = fn_args.get("query", question)
                try:
                    location = await run_segmentation_cb(query)
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": json.dumps({  # must be string, not dict
                            "summary": location.get("summary", ""),
                            "detections_count": len(location.get("detections", [])),
                        }),
                    })
                except Exception as exc:
                    logger.error("run_segmentation_cb raised: %s", exc)
                    # Same reason: plain string keeps the tool message valid.
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": f"Tool execution failed: {exc}",
                    })

            else:
                logger.warning("FunctionGemma called unknown tool: %s", fn_name)
                tool_results.append({
                    "name": fn_name,
                    "response": f"Unknown tool: {fn_name}",
                })

        # Append all tool results as a single tool turn.
        # Ollama's /api/chat endpoint requires "content" to be a string —
        # a list value is rejected with 400 at the API parser level before
        # the chat template even runs.  Serialise to JSON so the model can
        # still read the structured result while keeping content a string.
        messages.append({
            "role": "tool",
            "content": json.dumps(tool_results),
        })

        if should_finish:
            logger.info("FunctionGemma called finish. Exiting agentic loop.")
            break

    else:
        # Loop exhausted without a clean break
        logger.warning(
            "Agentic loop reached maximum iterations (%d). Stopping.", MAX_LOOP_ITERATIONS
        )

    # Fallback: if route_decided was never emitted (e.g. first call failed),
    # emit a default so the SSE stream stays consistent.
    if not route_decided_emitted:
        await emit("route_decided", "Route: analyze_diagnosis")

    return {
        "diagnosis": diagnosis,
        "location": location,
        "final_text": final_text,
    }
