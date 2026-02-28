from __future__ import annotations

from typing import Any


def _get_labeled_box(boxes: list[dict[str, Any]], label: str) -> dict[str, Any] | None:
    label_lower = label.lower()
    for box in boxes:
        box_label = str(box.get("label", "")).strip().lower()
        if box_label == label_lower:
            return box
    return None


def _box_height(box: dict[str, Any]) -> float:
    return float(box["y2"]) - float(box["y1"])


def _box_width(box: dict[str, Any]) -> float:
    return float(box["x2"]) - float(box["x1"])


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def run_cdr_calculator(paligemma_output: dict[str, Any]) -> dict[str, Any]:
    boxes = paligemma_output.get("boxes", [])
    if not isinstance(boxes, list):
        return {
            "tool": "cdr_calculator",
            "error": "paligemma boxes missing or invalid",
        }

    disc_box = _get_labeled_box(boxes, "optic-disc")
    cup_box = _get_labeled_box(boxes, "optic-cup")
    if not disc_box or not cup_box:
        return {
            "tool": "cdr_calculator",
            "error": "required optic-disc and optic-cup boxes were not found",
        }

    disc_h = _box_height(disc_box)
    disc_w = _box_width(disc_box)
    cup_h = _box_height(cup_box)
    cup_w = _box_width(cup_box)

    vertical_cdr = _safe_ratio(cup_h, disc_h)
    horizontal_cdr = _safe_ratio(cup_w, disc_w)
    area_ratio = _safe_ratio(cup_h * cup_w, disc_h * disc_w)

    result: dict[str, Any] = {
        "tool": "cdr_calculator",
        "input_source": "paligemma.boxes",
        "disc_box": disc_box,
        "cup_box": cup_box,
        "disc_height_px": round(disc_h, 2),
        "disc_width_px": round(disc_w, 2),
        "cup_height_px": round(cup_h, 2),
        "cup_width_px": round(cup_w, 2),
        "vertical_cdr": None if vertical_cdr is None else round(vertical_cdr, 4),
        "horizontal_cdr": None if horizontal_cdr is None else round(horizontal_cdr, 4),
        "area_cup_disc_ratio": None if area_ratio is None else round(area_ratio, 4),
    }

    if vertical_cdr is not None:
        if vertical_cdr >= 0.7:
            cdr_band = "high"
        elif vertical_cdr >= 0.6:
            cdr_band = "borderline_high"
        else:
            cdr_band = "not_high"
        result["vertical_cdr_band"] = cdr_band

    return result
