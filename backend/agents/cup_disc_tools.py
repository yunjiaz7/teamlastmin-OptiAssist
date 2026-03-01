from __future__ import annotations

"""
cup_disc_tools.py

Pure computation tools for cup-to-disc ratio (CDR) metrics.

Each function accepts the raw ``detections`` list produced by PaliGemma 2 and
returns a structured result dict with the computed metric and a brief clinical
interpretation.

PaliGemma detection format
--------------------------
detections: list of
    {
        "label":        str,   # e.g. "optic-disc", "optic-cup"
        "bounding_box": {
            "x_min": int, "y_min": int,
            "x_max": int, "y_max": int,
        }
    }

CDR reference ranges (approximate clinical thresholds)
-------------------------------------------------------
  vCDR / hCDR  ≤ 0.5   → Normal
  vCDR / hCDR  0.5–0.7 → Borderline
  vCDR / hCDR  > 0.7   → Elevated (suspect glaucoma)
  area CDR     ≤ 0.30  → Normal
  area CDR     0.30–0.50 → Borderline
  area CDR     > 0.50  → Elevated
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _box_area(box: dict) -> int:
    """Return the pixel area of a bounding box dict."""
    return (box["x_max"] - box["x_min"]) * (box["y_max"] - box["y_min"])


def _extract_boxes(
    detections: list[dict],
) -> tuple[Optional[dict], Optional[dict]]:
    """
    Extract optic disc and optic cup bounding boxes from a PaliGemma
    detections list.

    Three strategies are tried in order:

    1. **Label matching** — case-insensitive scan for "disc"/"disk" and "cup".
       Works when PaliGemma outputs readable text labels between loc tokens.

    2. **Area-based fallback** — if labels are non-descriptive ("region_1",
       "region_2", empty strings), assign by area: the optic disc is always
       the physically larger structure, so largest box → disc, second
       largest → cup.  This is the most reliable anatomical heuristic.

    3. **Index-based fallback** — PaliGemma generates boxes in prompt order.
       The detection prompt is ``"detect optic-disc ; optic-cup"``, so
       index 0 → disc, index 1 → cup.  Used only as a last resort.

    Args:
        detections: List of detection dicts produced by PaliGemma 2, each
            with keys ``"label"`` and ``"bounding_box"``.

    Returns:
        (disc_box, cup_box) — each is a ``{x_min, y_min, x_max, y_max}``
        dict, or ``None`` if the structure cannot be determined.
    """
    disc_box: Optional[dict] = None
    cup_box: Optional[dict] = None

    # ------------------------------------------------------------------
    # Strategy 1: semantic label matching
    # ------------------------------------------------------------------
    for det in detections:
        raw_label = det.get("label", "")
        label = raw_label.lower().replace("_", "-").replace(" ", "-")
        box = det.get("bounding_box") or {}
        if not box:
            continue
        # Cup check before disc so "optic-cup" isn't swallowed by "disc"
        if "cup" in label:
            cup_box = box
        elif "disc" in label or "disk" in label:
            disc_box = box

    if disc_box is not None and cup_box is not None:
        return disc_box, cup_box

    # ------------------------------------------------------------------
    # Strategy 2: area-based fallback
    # Optic disc > optic cup anatomically.  Sort by area and assign.
    # ------------------------------------------------------------------
    valid_boxes = [
        det.get("bounding_box")
        for det in detections
        if det.get("bounding_box")
    ]
    valid_boxes = [b for b in valid_boxes if b]  # drop None/empty

    if len(valid_boxes) >= 2:
        sorted_by_area = sorted(valid_boxes, key=_box_area, reverse=True)
        if disc_box is None:
            disc_box = sorted_by_area[0]
            logger.debug(
                "Disc assigned by area fallback (largest box, area=%d)",
                _box_area(disc_box),
            )
        if cup_box is None:
            # Pick the largest remaining box that is smaller than disc
            for box in sorted_by_area[1:]:
                if _box_area(box) < _box_area(disc_box):
                    cup_box = box
                    logger.debug(
                        "Cup assigned by area fallback (area=%d)",
                        _box_area(cup_box),
                    )
                    break

    if disc_box is not None and cup_box is not None:
        return disc_box, cup_box

    # ------------------------------------------------------------------
    # Strategy 3: index-based fallback (prompt order: disc first, cup second)
    # ------------------------------------------------------------------
    if len(valid_boxes) >= 1 and disc_box is None:
        disc_box = valid_boxes[0]
        logger.debug("Disc assigned by index fallback (index 0)")
    if len(valid_boxes) >= 2 and cup_box is None:
        cup_box = valid_boxes[1]
        logger.debug("Cup assigned by index fallback (index 1)")

    if disc_box is None or cup_box is None:
        labels = [d.get("label", "<empty>") for d in detections]
        logger.warning(
            "Could not assign disc/cup boxes from %d detections. labels=%s",
            len(detections), labels,
        )

    return disc_box, cup_box


def _missing(metric: str, what: str) -> dict:
    """Return a standardised error result when a required structure is absent."""
    return {
        "metric": metric,
        "value": None,
        "error": f"{what} bounding box not found in detections.",
    }


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def compute_vertical_cdr(detections: list[dict]) -> dict:
    """
    Compute the vertical cup-to-disc ratio (vCDR).

    Formula:  vCDR = cup_height / disc_height
              where height = y_max − y_min.

    Args:
        detections: PaliGemma detection list (must contain optic-disc and
                    optic-cup entries).

    Returns:
        Dict with keys: metric, value, cup_height_px, disc_height_px,
        interpretation.  ``value`` is None on error.
    """
    disc_box, cup_box = _extract_boxes(detections)

    if disc_box is None:
        return _missing("vertical_cdr", "Optic disc")
    if cup_box is None:
        return _missing("vertical_cdr", "Optic cup")

    disc_h = disc_box["y_max"] - disc_box["y_min"]
    cup_h = cup_box["y_max"] - cup_box["y_min"]

    if disc_h == 0:
        return {"metric": "vertical_cdr", "value": None, "error": "Disc height is zero."}

    vcdr = round(cup_h / disc_h, 3)
    if vcdr <= 0.5:
        interpretation = "Normal"
    elif vcdr <= 0.7:
        interpretation = "Borderline — monitor closely"
    else:
        interpretation = "Elevated — suspect glaucoma"

    logger.info("vertical_cdr=%.3f  disc_h=%d  cup_h=%d", vcdr, disc_h, cup_h)
    return {
        "metric": "vertical_cdr",
        "value": vcdr,
        "cup_height_px": cup_h,
        "disc_height_px": disc_h,
        "interpretation": interpretation,
    }


def compute_horizontal_cdr(detections: list[dict]) -> dict:
    """
    Compute the horizontal cup-to-disc ratio (hCDR).

    Formula:  hCDR = cup_width / disc_width
              where width = x_max − x_min.

    Args:
        detections: PaliGemma detection list.

    Returns:
        Dict with keys: metric, value, cup_width_px, disc_width_px,
        interpretation.
    """
    disc_box, cup_box = _extract_boxes(detections)

    if disc_box is None:
        return _missing("horizontal_cdr", "Optic disc")
    if cup_box is None:
        return _missing("horizontal_cdr", "Optic cup")

    disc_w = disc_box["x_max"] - disc_box["x_min"]
    cup_w = cup_box["x_max"] - cup_box["x_min"]

    if disc_w == 0:
        return {"metric": "horizontal_cdr", "value": None, "error": "Disc width is zero."}

    hcdr = round(cup_w / disc_w, 3)
    if hcdr <= 0.5:
        interpretation = "Normal"
    elif hcdr <= 0.7:
        interpretation = "Borderline — monitor closely"
    else:
        interpretation = "Elevated — suspect glaucoma"

    logger.info("horizontal_cdr=%.3f  disc_w=%d  cup_w=%d", hcdr, disc_w, cup_w)
    return {
        "metric": "horizontal_cdr",
        "value": hcdr,
        "cup_width_px": cup_w,
        "disc_width_px": disc_w,
        "interpretation": interpretation,
    }


def compute_area_cdr(detections: list[dict]) -> dict:
    """
    Compute the area-based cup-to-disc ratio.

    Bounding-box areas are used as proxies for the true optic-disc and
    optic-cup areas.

    Formula:  area_CDR = cup_area / disc_area

    Args:
        detections: PaliGemma detection list.

    Returns:
        Dict with keys: metric, value, cup_area_px2, disc_area_px2,
        interpretation.
    """
    disc_box, cup_box = _extract_boxes(detections)

    if disc_box is None:
        return _missing("area_cdr", "Optic disc")
    if cup_box is None:
        return _missing("area_cdr", "Optic cup")

    disc_area = (
        (disc_box["x_max"] - disc_box["x_min"])
        * (disc_box["y_max"] - disc_box["y_min"])
    )
    cup_area = (
        (cup_box["x_max"] - cup_box["x_min"])
        * (cup_box["y_max"] - cup_box["y_min"])
    )

    if disc_area == 0:
        return {"metric": "area_cdr", "value": None, "error": "Disc area is zero."}

    area_cdr = round(cup_area / disc_area, 3)
    if area_cdr <= 0.30:
        interpretation = "Normal"
    elif area_cdr <= 0.50:
        interpretation = "Borderline — monitor closely"
    else:
        interpretation = "Elevated — suspect glaucoma"

    logger.info("area_cdr=%.3f  disc_area=%d  cup_area=%d", area_cdr, disc_area, cup_area)
    return {
        "metric": "area_cdr",
        "value": area_cdr,
        "cup_area_px2": cup_area,
        "disc_area_px2": disc_area,
        "interpretation": interpretation,
    }


def compute_disc_diameter(detections: list[dict]) -> dict:
    """
    Estimate the optic disc diameter in pixels.

    Returns both the width and height of the bounding box and their mean
    as the average diameter.

    Args:
        detections: PaliGemma detection list.

    Returns:
        Dict with keys: metric, value_px (mean diameter), disc_width_px,
        disc_height_px, bounding_box.
    """
    disc_box, _ = _extract_boxes(detections)

    if disc_box is None:
        return _missing("disc_diameter", "Optic disc")

    width = disc_box["x_max"] - disc_box["x_min"]
    height = disc_box["y_max"] - disc_box["y_min"]
    avg_diameter = round((width + height) / 2.0, 1)

    logger.info("disc_diameter  w=%d  h=%d  avg=%.1f", width, height, avg_diameter)
    return {
        "metric": "disc_diameter",
        "value_px": avg_diameter,
        "disc_width_px": width,
        "disc_height_px": height,
        "bounding_box": disc_box,
    }


def compute_cup_diameter(detections: list[dict]) -> dict:
    """
    Estimate the optic cup diameter in pixels.

    Returns both the width and height of the bounding box and their mean
    as the average diameter.

    Args:
        detections: PaliGemma detection list.

    Returns:
        Dict with keys: metric, value_px (mean diameter), cup_width_px,
        cup_height_px, bounding_box.
    """
    _, cup_box = _extract_boxes(detections)

    if cup_box is None:
        return _missing("cup_diameter", "Optic cup")

    width = cup_box["x_max"] - cup_box["x_min"]
    height = cup_box["y_max"] - cup_box["y_min"]
    avg_diameter = round((width + height) / 2.0, 1)

    logger.info("cup_diameter  w=%d  h=%d  avg=%.1f", width, height, avg_diameter)
    return {
        "metric": "cup_diameter",
        "value_px": avg_diameter,
        "cup_width_px": width,
        "cup_height_px": height,
        "bounding_box": cup_box,
    }
