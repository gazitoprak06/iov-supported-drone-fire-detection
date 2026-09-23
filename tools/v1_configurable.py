"""
A parameterized superset of the released hand-set pipeline.

v1_heuristic.run_heuristic_pipeline is the reference implementation and every
experiment that reports the released configuration calls it directly. The
ablation of Section 4.2 and the saturation sweep of Section 4.4 need to switch
one colour-space mask off, or move the saturation floors, which the reference
function does not expose. Rather than let those experiments carry a private
copy of the pipeline, this module exposes the same computation with the two
parameters the experiments vary.

The equivalence is not asserted in prose: tools/verify_v1_equivalence.py runs
this module against the reference function over the whole test split at the
default configuration and fails if a single emitted box differs.

Two conventions, both following the manuscript:
  * a disabled colour space contributes an all-pass mask to the intersection,
    so disabling all three leaves only the white-hot bypass and the morphology;
  * the white-hot bypass is not one of the three colour spaces and stays active
    in every ablation row, because Section 3.1 introduces it as a correction to
    overexposure rather than as a chromatic filter.
"""

from __future__ import annotations

import cv2
import numpy as np

from v1_heuristic import get_ellipse_kernel

# Released saturation floors of the three hue bands (Section 3.1).
DEFAULT_SAT_FLOORS = (60, 80, 60)

# Released hue bands, in the 0-179 integer range OpenCV emits.
HUE_BANDS = ((0, 15), (15, 35), (160, 180))

ALL_SPACES = ("hsv", "rgb", "ycc")


def _all_pass(shape) -> np.ndarray:
    return np.full(shape, 255, dtype=np.uint8)


def build_masks(frame, active=ALL_SPACES, sat_floors=DEFAULT_SAT_FLOORS):
    """Return the three colour-space masks, with disabled ones all-pass."""
    shape = frame.shape[:2]

    if "hsv" in active:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        H, S, V = cv2.split(hsv)
        (lo0, hi0), (lo1, hi1), (lo2, hi2) = HUE_BANDS
        s0, s1, s2 = sat_floors
        m_hsv = (
            ((H >= lo0) & (H <= hi0) & (S >= s0))
            | ((H >= lo1) & (H <= hi1) & (S >= s1))
            | ((H >= lo2) & (H <= hi2) & (S >= s2))
        ) & (V >= 210)
        m_hsv = m_hsv.astype(np.uint8) * 255
    else:
        m_hsv = _all_pass(shape)

    B, G, R = cv2.split(frame)

    if "rgb" in active:
        m_rgb = ((R > G) & (G > B) & (R > 210) & (G > 130)).astype(np.uint8) * 255
    else:
        m_rgb = _all_pass(shape)

    if "ycc" in active:
        ycc = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        Y, Cr, Cb = cv2.split(ycc)
        m_ycc = ((Cr > 150) & (Cr > Cb) & (Y > Cb)).astype(np.uint8) * 255
    else:
        m_ycc = _all_pass(shape)

    return m_hsv, m_rgb, m_ycc, (B, G, R)


def run_configurable(
    frame,
    prev_frame=None,
    h: float = 50.0,
    active=ALL_SPACES,
    sat_floors=DEFAULT_SAT_FLOORS,
    whitehot: str = "always",
    stages: bool = False,
):
    """Algorithm 1 with the colour-space set and saturation floors exposed.

    With active=ALL_SPACES, sat_floors=DEFAULT_SAT_FLOORS and whitehot="always"
    this reproduces v1_heuristic.run_heuristic_pipeline exactly; see
    verify_v1_equivalence.py.

    whitehot selects how the overexposure bypass is treated in an ablation row.
    Section 3.1 introduces it separately from the three colour spaces, but it is
    itself an RGB-channel rule, so two readings are defensible and they give
    different numbers on the rows where the RGB mask is off:
      "always"      the bypass is not one of the three colour spaces and stays
                    active in every row;
      "follows_rgb" the bypass is part of the RGB stage and is disabled with it.

    stages=True additionally returns the box count at each stage of the
    localization chain, which Section 4.3 needs in order to state at which
    stage its spurious-box totals were counted.
    """
    m_hsv, m_rgb, m_ycc, (B, G, R) = build_masks(frame, active, sat_floors)

    M = cv2.bitwise_and(m_hsv, cv2.bitwise_and(m_rgb, m_ycc))

    kernel_15 = get_ellipse_kernel(15)
    use_wh = whitehot == "always" or (whitehot == "follows_rgb" and "rgb" in active)
    if use_wh:
        # White-hot bypass, eroded then merged BEFORE the morphological chain.
        m_wh = ((R > 240) & (G > 230) & (B > 200)).astype(np.uint8) * 255
        m_wh = cv2.erode(m_wh, kernel_15)
        M = cv2.bitwise_or(M, m_wh)

    M = cv2.morphologyEx(M, cv2.MORPH_CLOSE, kernel_15)
    M = cv2.morphologyEx(M, cv2.MORPH_OPEN, get_ellipse_kernel(5))
    M = cv2.dilate(M, kernel_15, iterations=2)

    m_flick = None
    if prev_frame is not None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(gray, prev_gray)
        _, diff_bin = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        m_flick = cv2.morphologyEx(diff_bin, cv2.MORPH_CLOSE, get_ellipse_kernel(7))

    A_min = 600.0 * (50.0 / max(1.0, float(h))) ** 2
    contours, _ = cv2.findContours(M, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for c in contours:
        area = cv2.contourArea(c)
        if area > A_min:
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            s = area / hull_area if hull_area > 0 else 0

            x, y, wb, hb = cv2.boundingRect(c)

            if m_flick is not None:
                roi = m_flick[y : y + hb, x : x + wb]
                rho = np.mean(roi) / 255.0
                delta = -0.30 if rho < 0.01 else 0.30 * rho
            else:
                delta = 0.0

            c_alg = min(1.0, 0.60 + 0.10 * s + delta)
            if c_alg >= 0.50:
                boxes.append({"rect": (x, y, wb, hb), "score": c_alg, "area": area})

    # Asymmetric NMS: erased only by a box of strictly greater area, single pass.
    surviving = []
    for i, bA in enumerate(boxes):
        keep = True
        xA, yA, wA, hA = bA["rect"]
        areaA = wA * hA
        for j, bB in enumerate(boxes):
            if i == j:
                continue
            xB, yB, wB, hB = bB["rect"]
            areaB = wB * hB
            if areaB > areaA:
                xi1, yi1 = max(xA, xB), max(yA, yB)
                xi2, yi2 = min(xA + wA, xB + wB), min(yA + hA, yB + hB)
                if xi2 > xi1 and yi2 > yi1:
                    inter = (xi2 - xi1) * (yi2 - yi1)
                    if inter / float(areaA) > 0.30:
                        keep = False
                        break
        if keep:
            surviving.append(bA)

    n_pre_nms = len(boxes)
    n_post_nms = len(surviving)

    # Proximity merge, iterated to a fixed point.
    d_merge = max(20.0, 100.0 - 0.5 * h)
    while True:
        merged = False
        new_boxes = []
        used = set()
        for i, bA in enumerate(surviving):
            if i in used:
                continue
            xA, yA, wA, hA = bA["rect"]
            merged_box = dict(bA)
            for j, bB in enumerate(surviving):
                if i == j or j in used:
                    continue
                xB, yB, wB, hB = bB["rect"]
                dx = max(0, max(xA - (xB + wB), xB - (xA + wA)))
                dy = max(0, max(yA - (yB + hB), yB - (yA + hA)))
                dist = np.sqrt(dx * dx + dy * dy)
                if dist <= d_merge:
                    x_new, y_new = min(xA, xB), min(yA, yB)
                    w_new = max(xA + wA, xB + wB) - x_new
                    h_new = max(yA + hA, yB + hB) - y_new
                    merged_box["rect"] = (x_new, y_new, w_new, h_new)
                    merged_box["score"] = max(merged_box["score"], bB["score"])
                    merged_box["area"] += bB["area"]
                    used.add(j)
                    merged = True
                    xA, yA, wA, hA = x_new, y_new, w_new, h_new
            used.add(i)
            new_boxes.append(merged_box)
        surviving = new_boxes
        if not merged:
            break

    if stages:
        return surviving, {
            "pre_nms": n_pre_nms,
            "post_nms": n_post_nms,
            "post_merge": len(surviving),
        }
    return surviving


# The seven configurations of Table 2, in the order the manuscript prints them.
ABLATION_CONFIGS = [
    ("HSV only", ("hsv",)),
    ("RGB only", ("rgb",)),
    ("YCbCr only", ("ycc",)),
    ("HSV + RGB", ("hsv", "rgb")),
    ("HSV + YCbCr", ("hsv", "ycc")),
    ("RGB + YCbCr", ("rgb", "ycc")),
    ("HSV + RGB + YCbCr (proposed)", ("hsv", "rgb", "ycc")),
]
