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

MAX_LOOP_ITERATIONS = 8

# ---------------------------------------------------------------------------
# BYPASS FLAG — set True to skip FunctionGemma and run both agents directly.
# Useful when FunctionGemma is unavailable or you want to force PaliGemma.
# The user's question is passed as-is to run_segmentation as the query.
# ---------------------------------------------------------------------------
BYPASS_FUNCTIONGEMMA = False

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

# ---------------------------------------------------------------------------
# Orchestration instructions
#
# These live in the USER message — NOT in the developer message — so they
# do not interfere with FunctionGemma's function-calling activation signal.
#
# Agent roles:
#   run_diagnosis    → MedGemma 4B (general medical model by Google).
#                      Analyzes the retinal fundus image; returns structured JSON
#                      with condition, severity, findings, recommendation.
#
#   run_segmentation → PaliGemma 2 (fine-tuned vision model by Google).
#                      Detects optic disc and optic cup bounding boxes from
#                      the fundus image. Output is passed directly to MedGemma
#                      for the final summary.
# ---------------------------------------------------------------------------
_ORCHESTRATION_INSTRUCTIONS = (
    "\n\n---\n"
    "MANDATORY RULES — follow exactly, no exceptions:\n"
    "1. Never answer directly. Only call tools.\n"
    "2. ALWAYS call run_diagnosis first.\n"
    "3. After run_diagnosis returns:\n"
    "   - If the question is about optic disc, optic cup, cup-to-disc ratio, "
    "CDR, glaucoma, or disc cupping: "
    "MUST call run_segmentation, then call finish.\n"
    "   - Otherwise: call finish.\n"
    "4. finish is always the last call.\n"
    "Start now: call run_diagnosis."
)

# Message injected when FunctionGemma answers in text instead of calling a tool
# (pre-diagnosis phase)
_NUDGE_MESSAGE = (
    "You responded with text instead of calling a tool. "
    "This is not allowed. "
    "You MUST call run_diagnosis right now. "
    "Do not output any text — only call the run_diagnosis tool."
)

# Message injected when FunctionGemma skips run_segmentation after diagnosis
# for CDR / optic-disc related questions.
_NUDGE_SEGMENTATION_MESSAGE = (
    "You responded with text instead of calling run_segmentation. "
    "This is not allowed. "
    "The question is about cup-to-disc ratio / optic disc / glaucoma. "
    "You MUST call run_segmentation right now — do NOT output text."
)

# Keywords that require PaliGemma segmentation after diagnosis.
_SEGMENTATION_KEYWORDS = frozenset({
    "optic disc", "optic cup", "cup to disc", "cup-to-disc",
    "cdr", "glaucoma", "disc cupping", "neuroretinal rim",
    "cup disc", "disc cup",
})

# ---------------------------------------------------------------------------
# Tool definitions exposed to FunctionGemma
# ---------------------------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_diagnosis",
            "description": (
                "Run MedGemma 4B — Google's general-purpose medical AI model — "
                "to analyze the retinal fundus image and produce a structured "
                "medical diagnosis. Returns condition, severity (None/Mild/Moderate/"
                "Severe/Proliferative), confidence score, a list of specific findings, "
                "and a clinical recommendation. Always call this tool first before "
                "any other tool."
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
                "Run PaliGemma 2 to detect optic disc and optic cup bounding boxes "
                "from the retinal fundus image. "
                "Call this whenever the question asks about: optic disc, optic cup, "
                "cup-to-disc ratio, CDR, glaucoma, disc cupping, or neuroretinal rim. "
                "Must be called after run_diagnosis and before finish."
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
            "name": "finish",
            "description": (
                "Signal that the analysis workflow is complete and all required "
                "tools have been called. MedGemma will then produce a final "
                "clinical narrative summarising all outputs. Always call this "
                "as the last step."
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


def _needs_segmentation(question: str) -> bool:
    """Return True if the question requires PaliGemma segmentation."""
    q = question.lower()
    return any(kw in q for kw in _SEGMENTATION_KEYWORDS)


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
        question:             The clinician's question.
        image_description:    Pre-scanned image description from the prescanner.
        run_diagnosis_cb:     Async callback that runs MedGemma diagnosis;
                              emits medgemma_start / medgemma_complete internally.
        run_segmentation_cb:  Async callback(query: str) that runs PaliGemma 2
                              segmentation; emits paligemma_start / paligemma_complete
                              internally.
        emit:                 Async SSE emit callback — called as
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

    # ----------------------------------------------------------------
    # BYPASS — skip FunctionGemma and run both agents unconditionally.
    # Remove BYPASS_FUNCTIONGEMMA (or set it False) to restore normal routing.
    # ----------------------------------------------------------------
    if BYPASS_FUNCTIONGEMMA:
        logger.info("BYPASS mode: skipping FunctionGemma AND MedGemma, running PaliGemma only.")
        await emit("route_decided", "Route: analyze_location")

        location: dict | None = None
        seg_query = "detect optic-disc ; optic-cup"  # fixed prompt PaliGemma was fine-tuned on
        try:
            location = await run_segmentation_cb(seg_query)
        except Exception as exc:
            logger.error("bypass run_segmentation_cb raised: %s", exc)

        return {
            "diagnosis": None,
            "location": location,
            "final_text": "",
        }

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
        # Phase 1 (pre-diagnosis): nudge toward run_diagnosis.
        # Phase 2 (post-diagnosis, CDR question): nudge toward run_segmentation.
        # ----------------------------------------------------------------
        if not tool_calls:
            if diagnosis is None and nudge_count < MAX_NUDGE_RETRIES:
                # Phase 1 nudge: diagnosis hasn't run yet.
                nudge_count += 1
                logger.warning(
                    "FunctionGemma returned text without calling a tool "
                    "(iteration %d, nudge %d/%d). Injecting reminder.",
                    iteration + 1,
                    nudge_count,
                    MAX_NUDGE_RETRIES,
                )
                if text_content:
                    messages.append({"role": "assistant", "content": text_content})
                messages.append({"role": "user", "content": _NUDGE_MESSAGE})
                continue  # retry this iteration slot

            if (
                diagnosis is not None
                and location is None
                and _needs_segmentation(question)
                and nudge_count < MAX_NUDGE_RETRIES
            ):
                # Phase 2 nudge: diagnosis ran but segmentation was skipped for
                # a CDR / optic-disc related question.
                nudge_count += 1
                logger.warning(
                    "FunctionGemma skipped run_segmentation for a CDR question "
                    "(iteration %d, nudge %d/%d). Injecting segmentation reminder.",
                    iteration + 1,
                    nudge_count,
                    MAX_NUDGE_RETRIES,
                )
                if text_content:
                    messages.append({"role": "assistant", "content": text_content})
                messages.append({"role": "user", "content": _NUDGE_SEGMENTATION_MESSAGE})
                continue  # retry this iteration slot

            # Out of nudge budget or no segmentation required → accept text exit.
            final_text = text_content
            logger.info(
                "FunctionGemma returned text (no tool calls). "
                "nudge_count=%d diagnosis_done=%s segmentation_done=%s. Exiting loop.",
                nudge_count,
                diagnosis is not None,
                location is not None,
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

            if not fn_name:
                logger.warning("FunctionGemma returned a tool_call with an empty function name; skipping.")
                continue

            logger.info(
                "FunctionGemma called tool=%s args=%s", fn_name, fn_args
            )

            # Emit route_decided once, based on the first meaningful tool call
            if not route_decided_emitted and fn_name in _TOOL_TO_ROUTE:
                route_name = _TOOL_TO_ROUTE[fn_name]
                await emit("route_decided", f"Route: {route_name}")
                logger.info("Route decided: %s", route_name)
                route_decided_emitted = True

            # ----------------------------------------------------------
            if fn_name == "finish":
                should_finish = True
                tool_results.append({"name": "finish", "response": "Analysis complete."})

            # ----------------------------------------------------------
            elif fn_name == "run_diagnosis":
                try:
                    diagnosis = await run_diagnosis_cb()
                    tool_results.append({
                        "name": "run_diagnosis",
                        "response": json.dumps(diagnosis),
                    })
                except Exception as exc:
                    logger.error("run_diagnosis_cb raised: %s", exc)
                    tool_results.append({
                        "name": "run_diagnosis",
                        "response": f"Tool execution failed: {exc}",
                    })

            # ----------------------------------------------------------
            elif fn_name == "run_segmentation":
                # PaliGemma 2 is fine-tuned specifically for this prompt.
                # The tool takes no arguments — the prompt is always fixed.
                query: str = "detect optic-disc ; optic-cup"
                try:
                    location = await run_segmentation_cb(query)
                    detections_out = location.get("detections", [])
                    # Log raw output and labels to aid debugging label parsing
                    raw_out = location.get("raw_output", "")[:200]
                    labels_found = [d.get("label", "<empty>") for d in detections_out]
                    logger.info(
                        "PaliGemma detections=%d  labels=%s  raw_output=%r",
                        len(detections_out), labels_found, raw_out,
                    )
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": json.dumps({
                            "summary": location.get("summary", ""),
                            "detections_count": len(detections_out),
                            "detections": detections_out,
                        }),
                    })
                except (FileNotFoundError, ImportError) as exc:
                    # Model is permanently unavailable (missing files or missing
                    # dependency) — retrying will not help. Signal to FunctionGemma
                    # to skip segmentation and call finish immediately.
                    logger.error("run_segmentation_cb: model permanently unavailable, skipping: %s", exc)
                    location = None
                    should_finish = True
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": (
                            "Segmentation model is permanently unavailable. "
                            "Skip this tool and call finish immediately."
                        ),
                    })
                except Exception as exc:
                    logger.error("run_segmentation_cb raised: %s", exc)
                    tool_results.append({
                        "name": "run_segmentation",
                        "response": f"Tool execution failed: {exc}",
                    })

            # ----------------------------------------------------------
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
