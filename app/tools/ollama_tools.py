from __future__ import annotations

import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import ollama

DEFAULT_CONFIG = {
    "ollama_host": "http://127.0.0.1:11434",
    "medgemma_model": "bentplau/medgemma1.5-4b-it",
    "funcgemma_model": "funcgemma",
}
CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


def _load_config() -> dict[str, str]:
    config_data = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            loaded = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                for key in ("ollama_host", "medgemma_model", "funcgemma_model"):
                    value = loaded.get(key)
                    if isinstance(value, str) and value.strip():
                        config_data[key] = value.strip()
        except Exception:
            pass
    return config_data


_CONFIG = _load_config()
DEFAULT_OLLAMA_HOST = _CONFIG["ollama_host"]
MEDGEMMA_MODEL = _CONFIG["medgemma_model"]
FUNCGEMMA_MODEL = _CONFIG["funcgemma_model"]

DETECTION_HINTS = (
    "optic disc",
    "optic cup",
    "disc",
    "cup",
    "segment",
    "localize",
    "locate",
    "bbox",
    "boundary",
    "where",
)
CDR_HINTS = ("cdr", "cup-to-disc", "cup to disc", "cup-disc", "cup/disc", "ratio")
INTERPRETATION_HINTS = (
    "diagnosis",
    "risk",
    "impression",
    "severity",
    "interpret",
    "diabetic retinopathy",
    "retinopathy",
    "dr",
    "glaucoma suspect",
    "normal",
    "abnormal",
)


def _client() -> ollama.Client:
    return ollama.Client(host=DEFAULT_OLLAMA_HOST)


def _image_payload_path(image_path: str | Path) -> str:
    return str(Path(image_path).resolve())


def _image_payload_bytes(image_path: str | Path) -> bytes:
    return Path(image_path).read_bytes()


def _sanitize_text(text: str) -> str:
    return (text or "").strip()


def _intent_scores(query: str) -> tuple[int, int, int]:
    query_lower = query.lower()
    detection_score = sum(1 for token in DETECTION_HINTS if token in query_lower)
    cdr_score = sum(1 for token in CDR_HINTS if token in query_lower)
    interpretation_score = sum(1 for token in INTERPRETATION_HINTS if token in query_lower)
    return detection_score, cdr_score, interpretation_score


def _enforce_tool_policy(selected_tools: list[str], available_tools: list[str], query: str) -> list[str]:
    available = set(available_tools)
    normalized = [tool for tool in selected_tools if tool in available]
    detection_score, cdr_score, interpretation_score = _intent_scores(query=query)

    if cdr_score > 0:
        normalized = [tool for tool in normalized if tool != "paligemma" and tool != "cdr_calculator"]
        normalized = ["paligemma", "cdr_calculator", *normalized]
        if interpretation_score > 0 and "medgemma" in available and "medgemma" not in normalized:
            normalized.append("medgemma")
    elif detection_score > 0 and interpretation_score == 0:
        normalized = [tool for tool in normalized if tool != "medgemma"]
        normalized = ["paligemma", *normalized]
    elif interpretation_score > 0 and detection_score == 0:
        normalized = [tool for tool in normalized if tool != "paligemma" and tool != "cdr_calculator"]
        normalized = ["medgemma", *normalized]
    elif detection_score == 0 and cdr_score == 0 and interpretation_score == 0:
        normalized = [tool for tool in normalized if tool == "medgemma"]
        if "medgemma" in available and "medgemma" not in normalized:
            normalized = ["medgemma"]

    deduped: list[str] = []
    seen: set[str] = set()
    for tool in normalized:
        if tool in available and tool not in seen:
            deduped.append(tool)
            seen.add(tool)

    if not deduped and "medgemma" in available:
        deduped = ["medgemma"]
    return deduped


def _resolve_image_path(image_path: str | Path) -> Path:
    path_obj = Path(image_path).expanduser().resolve()
    if not path_obj.exists():
        raise FileNotFoundError(f"Image path does not exist: {path_obj}")
    if not path_obj.is_file():
        raise ValueError(f"Image path is not a file: {path_obj}")
    return path_obj


def _is_image_input_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "missing data required for image input" in text or ("image input" in text and "status code: 500" in text)


def _model_supports_vision(model: str) -> bool | None:
    try:
        metadata = _client().show(model)
    except Exception:
        return None

    capabilities = metadata.get("capabilities")
    if isinstance(capabilities, list):
        lowered = {str(item).lower() for item in capabilities}
        if "vision" in lowered:
            return True
        if lowered and "vision" not in lowered:
            return False

    serialized = json.dumps(metadata, ensure_ascii=True).lower()
    if any(token in serialized for token in ("vision_tower", "image", "multimodal", "siglip", "clip")):
        return True
    return None


def _extract_first_json_object(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    stripped = text.strip()
    try:
        parsed = json.loads(stripped)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", stripped, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        return None
    return None


def plan_tool_calls(query: str, available_tools: list[str]) -> dict[str, Any]:
    system_prompt = (
        "You are a tool-planning assistant for a fundus-image research workflow. "
        "Tool definitions: "
        "paligemma=optic disc/cup segmentation and localization tool that returns boxes/tokens, "
        "medgemma=fundus vision-language analysis model for natural-language interpretation, "
        "cdr_calculator=deterministic cup-disc ratio from paligemma boxes. "
        "Routing rules: "
        "queries about optic disc, optic cup, cup-disc ratio, or glaucoma structure should include paligemma; "
        "ratio/CDR terms should include cdr_calculator; "
        "use medgemma for descriptive interpretation questions over fundus images. "
        "Choose only from available tool ids. "
        "Return STRICT JSON only with keys: selected_tools, reason. "
        "selected_tools must be a JSON array of tool ids."
    )
    user_payload = {
        "query": query,
        "available_tools": available_tools,
        "selection_rules": [
            "Use paligemma for localization/detection/box evidence.",
            "Use cdr_calculator when query asks for cup-disc ratio, CDR, or cup vs disc size comparison.",
            "Use medgemma for interpretation, risk, impression, uncertainty explanation, or next checks.",
        ],
        "output_schema": {
            "selected_tools": ["tool_id"],
            "reason": "string",
        },
    }
    response = _client().chat(
        model=FUNCGEMMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
    )
    raw_text = _sanitize_text(response["message"]["content"])
    parsed = _extract_first_json_object(raw_text) or {}
    selected = parsed.get("selected_tools", [])
    reason = parsed.get("reason", "")

    normalized = [str(t).strip() for t in selected if str(t).strip() in set(available_tools)]
    normalized = _enforce_tool_policy(selected_tools=normalized, available_tools=available_tools, query=query)
    if normalized:
        return {
            "selected_tools": normalized,
            "reason": str(reason).strip() or "Model tool planner selected tools.",
            "planner_model": FUNCGEMMA_MODEL,
        }

    detection_score, cdr_score, interpretation_score = _intent_scores(query=query)
    fallback: list[str] = []
    if cdr_score > 0:
        fallback.extend(["paligemma", "cdr_calculator"])
        if interpretation_score > 0:
            fallback.append("medgemma")
    elif detection_score > 0 and interpretation_score == 0:
        fallback.append("paligemma")
    elif detection_score > 0 and interpretation_score > 0:
        fallback.extend(["paligemma", "medgemma"])
    else:
        fallback.append("medgemma")

    deduped = []
    seen: set[str] = set()
    for tool in fallback:
        if tool in available_tools and tool not in seen:
            deduped.append(tool)
            seen.add(tool)
    return {
        "selected_tools": deduped,
        "reason": "Fallback heuristic selected tools.",
        "planner_model": FUNCGEMMA_MODEL,
    }


def run_medgemma_vqa(image_path: str, query: str) -> dict[str, Any]:
    image_path_obj = _resolve_image_path(image_path)
    vision_support = _model_supports_vision(MEDGEMMA_MODEL)
    if vision_support is False:
        raise RuntimeError(
            f"Configured MEDGEMMA_MODEL '{MEDGEMMA_MODEL}' does not advertise vision support. "
            "Use a vision-capable tag such as 'bentplau/medgemma1.5-4b-it'."
        )

    image_path_payload = _image_payload_path(image_path_obj)
    image_bytes_payload = _image_payload_bytes(image_path_obj)
    image_b64_payload = base64.b64encode(image_bytes_payload).decode("ascii")
    system_prompt = (
        "You are an expert at analyzing fundus images for research workflows. "
        "You must only report findings supported by the image and user query. "
        "Do not fabricate lesions, measurements, or diagnoses. "
        "If visual evidence is weak, explicitly say the uncertainty and what additional evaluation is needed. "
        "Keep language precise, concise, and non-alarmist."
    )
    user_prompt = (
        f"Clinical question: {query}\n\n"
        "Return exactly these sections:\n"
        "1) Findings\n"
        "2) Interpretation\n"
        "3) Confidence (low|medium|high)\n"
        "4) Recommended next checks"
    )

    payload_candidates: list[str | bytes] = [image_path_payload, image_bytes_payload, image_b64_payload]
    response: dict[str, Any] | None = None
    last_image_error: Exception | None = None
    for payload in payload_candidates:
        try:
            response = _client().chat(
                model=MEDGEMMA_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt, "images": [payload]},
                ],
            )
            break
        except Exception as exc:
            if not _is_image_input_error(exc):
                raise
            last_image_error = exc
            continue

    if response is None:
        if last_image_error is None:
            raise RuntimeError("MedGemma request failed without a captured response or error.")
        raise RuntimeError(
            f"Failed to submit image input to model '{MEDGEMMA_MODEL}' using path/bytes/base64 payloads. "
            "Ensure Ollama serves a vision-capable MedGemma model (for example 'bentplau/medgemma1.5-4b-it'). "
            f"Last error: {last_image_error}"
        ) from last_image_error

    text = _sanitize_text(response["message"]["content"])
    return {
        "tool": "medgemma_vqa",
        "model": MEDGEMMA_MODEL,
        "query": query,
        "response": text,
    }


def run_medgemma_final_synthesis(query: str, tool_outputs: dict[str, Any]) -> dict[str, Any]:
    system_prompt = (
        "You are an expert at analyzing fundus images and writing final answers for a research prototype. "
        "Use only the provided tool outputs as evidence. "
        "Do not refuse if numeric/tool evidence is available; summarize it directly. "
        "Do not invent findings and do not claim diagnosis certainty."
    )
    user_payload = {
        "query": query,
        "tool_outputs": tool_outputs,
        "required_format": [
            "Direct answer to the user query in 2-4 sentences",
            "If cdr_calculator exists, include vertical_cdr, horizontal_cdr, and area_cup_disc_ratio explicitly",
        ],
    }
    response = _client().chat(
        model=MEDGEMMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
    )
    text = _sanitize_text(response["message"]["content"])
    return {
        "tool": "medgemma_final_synthesizer",
        "model": MEDGEMMA_MODEL,
        "response": text,
    }


def run_funcgemma_synthesis(query: str, tool_outputs: dict[str, Any]) -> dict[str, Any]:
    system_prompt = (
        "You are a clinical evidence synthesis assistant. Combine outputs from other tools into a conservative summary. "
        "Never invent findings and never hide conflicts across tools. "
        "Use only evidence present in the provided tool outputs. "
        "If evidence quality is limited, reduce confidence and say what is missing. "
        "Do not provide definitive diagnosis or treatment plans from this input alone. "
        "Never output this exact sentence: "
        "'Decision support only; not a diagnosis. Use clinical judgment and standard of care.'"
    )
    user_payload = {
        "query": query,
        "tool_outputs": tool_outputs,
        "output_format": {
            "final_answer": "string",
            "confidence": "low|medium|high",
            "conflicts": ["string"],
            "safety_notes": ["string"],
        },
    }

    response = _client().chat(
        model=FUNCGEMMA_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=True)},
        ],
    )

    text = _sanitize_text(response["message"]["content"])
    return {
        "tool": "funcgemma_synthesizer",
        "model": FUNCGEMMA_MODEL,
        "response": text,
    }
