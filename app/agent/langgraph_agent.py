from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.tools.deterministic_tools import run_cdr_calculator
from app.tools.ollama_tools import plan_tool_calls, run_medgemma_final_synthesis, run_medgemma_vqa
from app.tools.paligemma_tool import run_paligemma_detection


class AgentState(TypedDict, total=False):
    query: str
    image_path: str
    selected_tools: list[str]
    tool_plan_reason: str
    tool_outputs: dict[str, Any]
    synthesis_output: dict[str, Any]
    final_answer: str
    confidence: str
    errors: list[str]


DETECTION_HINTS = {
    "detect",
    "lesion",
    "optic disc",
    "cup",
    "bbox",
    "where",
    "location",
    "segment",
}
CDR_HINTS = {
    "cdr",
    "cup-to-disc",
    "cup to disc",
    "cup-disc",
    "cup/disc",
    "ratio",
}
INTERPRETATION_HINTS = {
    "diagnosis",
    "risk",
    "differential",
    "treatment",
    "management",
    "severity",
    "impression",
    "diabetic retinopathy",
    "retinopathy",
    "dr",
    "normal",
    "abnormal",
}
DETECTION_PROMPT = "detect optic-disc ; optic-cup"
AVAILABLE_TOOLS = ["paligemma", "medgemma", "cdr_calculator"]


def _requires_structural_tools(query: str) -> bool:
    query_lower = query.lower()
    detection_score = sum(1 for token in DETECTION_HINTS if token in query_lower)
    cdr_score = sum(1 for token in CDR_HINTS if token in query_lower)
    return (detection_score + cdr_score) > 0


def _router(state: AgentState) -> AgentState:
    query = state["query"]
    errors = list(state.get("errors", []))
    selected_tools: list[str] = []
    reason = ""
    query_lower = query.lower()

    try:
        plan = plan_tool_calls(query=query, available_tools=AVAILABLE_TOOLS)
        selected_tools = list(plan.get("selected_tools", []))
        reason = str(plan.get("reason", "")).strip() or "Tool planner selected tools."
    except Exception as exc:  # pragma: no cover - runtime model errors
        errors.append(f"tool_planner_error: {exc}")
        detection_score = sum(1 for k in DETECTION_HINTS if k in query_lower)
        interpretation_score = sum(1 for k in INTERPRETATION_HINTS if k in query_lower)
        if detection_score > 0 and interpretation_score == 0:
            selected_tools = ["paligemma"]
            reason = "Fallback keyword router selected detection tool."
        elif interpretation_score > 0 and detection_score == 0:
            selected_tools = ["medgemma"]
            reason = "Fallback keyword router selected interpretation tool."
        else:
            selected_tools = ["paligemma", "medgemma"]
            reason = "Fallback keyword router selected mixed tool set."

    if "cdr_calculator" in selected_tools and "paligemma" not in selected_tools:
        selected_tools = ["paligemma", *selected_tools]
        reason = reason + " Added paligemma dependency for cdr_calculator."

    if not selected_tools:
        selected_tools = ["paligemma", "medgemma"]
        reason = "Planner returned no tools; defaulting to core tools."

    return {
        "selected_tools": selected_tools,
        "tool_plan_reason": reason,
        "tool_outputs": {},
        "errors": errors,
    }


def _run_tools(state: AgentState) -> AgentState:
    selected_tools = set(state.get("selected_tools", []))
    query = state["query"]
    image_path = state["image_path"]
    outputs = dict(state.get("tool_outputs", {}))
    errors = list(state.get("errors", []))
    requires_structural_tools = _requires_structural_tools(query)
    allow_paligemma = "paligemma" in selected_tools and requires_structural_tools

    if "paligemma" in selected_tools and not allow_paligemma:
        errors.append(
            "paligemma_skipped: query does not request structural/localization/CDR evidence; using interpretation tools only"
        )

    if allow_paligemma:
        try:
            outputs["paligemma"] = run_paligemma_detection(
                image_path=image_path,
                query_context=DETECTION_PROMPT,
            )
        except Exception as exc:  # pragma: no cover - runtime model errors
            errors.append(f"paligemma_tool_error: {exc}")

    if "medgemma" in selected_tools:
        try:
            outputs["medgemma"] = run_medgemma_vqa(image_path=image_path, query=query)
        except Exception as exc:  # pragma: no cover - runtime model errors
            errors.append(f"medgemma_tool_error: {exc}")

    if "cdr_calculator" in selected_tools and not requires_structural_tools:
        errors.append("cdr_calculator_skipped: query does not request cup-disc ratio analysis")
    elif "cdr_calculator" in selected_tools:
        try:
            if "paligemma" not in outputs:
                raise RuntimeError("cdr_calculator requires paligemma output but it is missing")
            outputs["cdr_calculator"] = run_cdr_calculator(outputs["paligemma"])
        except Exception as exc:  # pragma: no cover - runtime model errors
            errors.append(f"cdr_calculator_error: {exc}")

    if not outputs:
        errors.append("No tool produced output.")

    return {
        "tool_outputs": outputs,
        "errors": errors,
    }


def _normalize_synthesis_text(text: str) -> str:
    cleaned = (text or "").strip()
    if not cleaned:
        return cleaned
    if cleaned.startswith('"') and cleaned.endswith('"'):
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, str):
                return parsed.strip()
        except json.JSONDecodeError:
            pass
    return cleaned


def _deterministic_fallback_summary(query: str, tool_outputs: dict[str, Any]) -> str:
    cdr = tool_outputs.get("cdr_calculator")
    if not isinstance(cdr, dict):
        return ""
    if cdr.get("error"):
        return ""
    vertical = cdr.get("vertical_cdr")
    horizontal = cdr.get("horizontal_cdr")
    area_ratio = cdr.get("area_cup_disc_ratio")
    if vertical is None and horizontal is None and area_ratio is None:
        return ""
    return (
        f"Estimated cup-to-disc metrics from detected boxes: vertical CDR={vertical}, "
        f"horizontal CDR={horizontal}, area cup/disc ratio={area_ratio}. "
        "Correlate with full optic nerve head assessment and clinical context."
    )


def _paligemma_detection_summary(tool_outputs: dict[str, Any]) -> str:
    paligemma = tool_outputs.get("paligemma")
    if not isinstance(paligemma, dict):
        return ""
    box_count = paligemma.get("box_count")
    prediction = paligemma.get("prediction", "")
    return (
        f"Paligemma detection completed with {box_count} predicted boxes. "
        f"Predicted structures: {prediction}"
    )


def _synthesize(state: AgentState) -> AgentState:
    query = state["query"]
    tool_outputs = state.get("tool_outputs", {})
    errors = list(state.get("errors", []))

    try:
        synthesis = run_medgemma_final_synthesis(query=query, tool_outputs=tool_outputs)
        synthesis_text = _normalize_synthesis_text(synthesis["response"])
        confidence = "medium"
    except Exception as exc:  # pragma: no cover - runtime model errors
        errors.append(f"medgemma_synth_error: {exc}")
        synthesis = {"tool": "medgemma_final_synthesizer", "response": ""}
        synthesis_text = (
            "Synthesis model unavailable. Returning raw tool evidence only.\n\n"
            + json.dumps(tool_outputs, indent=2, ensure_ascii=True)
        )
        confidence = "low"

    if not synthesis_text or synthesis_text.lower().startswith("clinical support information"):
        deterministic_summary = _deterministic_fallback_summary(query=query, tool_outputs=tool_outputs)
        if deterministic_summary:
            synthesis_text = deterministic_summary
            confidence = "medium"

    if "cdr_calculator" not in tool_outputs and '"vertical_cdr": 0.0' in synthesis_text:
        paligemma_summary = _paligemma_detection_summary(tool_outputs=tool_outputs)
        if paligemma_summary:
            synthesis_text = paligemma_summary
            confidence = "low"

    return {
        "synthesis_output": synthesis,
        "final_answer": synthesis_text,
        "confidence": confidence,
        "errors": errors,
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", _router)
    graph.add_node("run_tools", _run_tools)
    graph.add_node("synthesize", _synthesize)
    graph.add_edge(START, "router")
    graph.add_edge("router", "run_tools")
    graph.add_edge("run_tools", "synthesize")
    graph.add_edge("synthesize", END)
    return graph.compile()


GRAPH = _build_graph()


def analyze_fundus_case(query: str, image_path: str) -> dict[str, Any]:
    if not Path(image_path).exists():
        raise FileNotFoundError(f"Image path does not exist: {image_path}")

    state: AgentState = {
        "query": query,
        "image_path": image_path,
    }
    result = GRAPH.invoke(state)
    return {
        "final_answer": result.get("final_answer", ""),
        "confidence": result.get("confidence", "low"),
        "decision_path": {
            "tool_plan_reason": result.get("tool_plan_reason", ""),
            "selected_tools": result.get("selected_tools", []),
            "tools_called": list(result.get("tool_outputs", {}).keys()),
        },
        "tool_outputs": result.get("tool_outputs", {}),
        "errors": result.get("errors", []),
    }
