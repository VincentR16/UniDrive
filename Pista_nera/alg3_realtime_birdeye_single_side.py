import argparse
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Result:
    ok: bool
    error_px: float | None
    overlay: np.ndarray
    bird_eye: np.ndarray | None = None
    used_single_side_estimation: bool = False
    estimation_mode: str = "none"


# ---------------------------------------------------------------------------
# Runtime bird-eye ROI controls
# ---------------------------------------------------------------------------
# In this version the bird-eye ROI is a draggable 4-point quadrilateral.
# Points are stored normalized in frame coordinates, so they keep working if
# the display is scaled. Order: 0=top-left, 1=top-right, 2=bottom-right, 3=bottom-left.
ROI_TRACKBAR_WINDOW = "Bird-eye / lane controls"
ROI_TRACKBARS_ACTIVE = False
MAIN_WINDOW_NAME = "ALG3 video realtime"

BIRD_CENTER_X_PERCENT = 50
BIRD_TOP_Y_PERCENT = 55
BIRD_BOTTOM_Y_PERCENT = 98
BIRD_TOP_WIDTH_PERCENT = 40
BIRD_BOTTOM_WIDTH_PERCENT = 100

BIRD_ROI_POINTS_NORM: list[list[float]] | None = None
BIRD_DRAGGING_POINT_IDX: int | None = None
BIRD_MOUSE_HOVER_POINT_IDX: int | None = None
LAST_FRAME_SHAPE: tuple[int, int, int] | None = None
MAIN_DISPLAY_SCALE = 1.0

# Optional realtime tuning of the expected road width used by the single-side estimator.
LANE_WIDTH_PERCENT_RUNTIME = 70


def put_text(img, lines, x=10, y=25, dy=24):
    out = img.copy()
    for i, txt in enumerate(lines):
        cv2.putText(
            out,
            txt,
            (x, y + i * dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return out


def roi_frame(frame, y_ratio=0.45):
    h, w = frame.shape[:2]
    y0 = int(h * y_ratio)
    return frame[y0:, :], y0


def largest_component(mask: np.ndarray):
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return None, None, None
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + int(np.argmax(areas))
    return labels == idx, stats[idx], cent[idx]


def _clip_int(value: int, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, value)))


def _default_roi_points_norm_from_percent(frame_shape) -> list[list[float]]:
    """
    Builds the initial quadrilateral from the old percentage parameters.
    Returned points are normalized [0, 1].
    """
    h, w = frame_shape[:2]
    center_x = w * BIRD_CENTER_X_PERCENT / 100.0
    top_y = h * BIRD_TOP_Y_PERCENT / 100.0
    bottom_y = h * BIRD_BOTTOM_Y_PERCENT / 100.0
    top_half_width = w * BIRD_TOP_WIDTH_PERCENT / 100.0 / 2.0
    bottom_half_width = w * BIRD_BOTTOM_WIDTH_PERCENT / 100.0 / 2.0

    pts = np.float32([
        [center_x - top_half_width, top_y],
        [center_x + top_half_width, top_y],
        [center_x + bottom_half_width, bottom_y],
        [center_x - bottom_half_width, bottom_y],
    ])

    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    norm = [[float(x / max(w - 1, 1)), float(y / max(h - 1, 1))] for x, y in pts]
    return _enforce_roi_point_order(norm)


def _enforce_roi_point_order(points_norm: list[list[float]]) -> list[list[float]]:
    """
    Keeps the 4 points usable for a perspective transform while still allowing
    each corner to be moved independently.
    """
    pts = [[float(np.clip(p[0], 0.0, 1.0)), float(np.clip(p[1], 0.0, 1.0))] for p in points_norm]
    min_sep = 0.01

    # Keep left/right order on top and bottom edges.
    if pts[0][0] > pts[1][0] - min_sep:
        mid = 0.5 * (pts[0][0] + pts[1][0])
        pts[0][0] = np.clip(mid - min_sep / 2, 0.0, 1.0)
        pts[1][0] = np.clip(mid + min_sep / 2, 0.0, 1.0)
    if pts[3][0] > pts[2][0] - min_sep:
        mid = 0.5 * (pts[3][0] + pts[2][0])
        pts[3][0] = np.clip(mid - min_sep / 2, 0.0, 1.0)
        pts[2][0] = np.clip(mid + min_sep / 2, 0.0, 1.0)

    # Keep upper corners above corresponding lower corners.
    if pts[0][1] > pts[3][1] - min_sep:
        mid = 0.5 * (pts[0][1] + pts[3][1])
        pts[0][1] = np.clip(mid - min_sep / 2, 0.0, 1.0)
        pts[3][1] = np.clip(mid + min_sep / 2, 0.0, 1.0)
    if pts[1][1] > pts[2][1] - min_sep:
        mid = 0.5 * (pts[1][1] + pts[2][1])
        pts[1][1] = np.clip(mid - min_sep / 2, 0.0, 1.0)
        pts[2][1] = np.clip(mid + min_sep / 2, 0.0, 1.0)

    return pts


def reset_birdeye_roi_points(frame_shape) -> None:
    """Reset the draggable quadrilateral to the percentage-based default."""
    global BIRD_ROI_POINTS_NORM, BIRD_DRAGGING_POINT_IDX, BIRD_MOUSE_HOVER_POINT_IDX
    BIRD_ROI_POINTS_NORM = _default_roi_points_norm_from_percent(frame_shape)
    BIRD_DRAGGING_POINT_IDX = None
    BIRD_MOUSE_HOVER_POINT_IDX = None


def _ensure_birdeye_roi_points(frame: np.ndarray) -> None:
    global BIRD_ROI_POINTS_NORM
    if BIRD_ROI_POINTS_NORM is None:
        reset_birdeye_roi_points(frame.shape)


def _roi_points_abs_from_norm(frame_shape) -> np.ndarray:
    global BIRD_ROI_POINTS_NORM
    if BIRD_ROI_POINTS_NORM is None:
        reset_birdeye_roi_points(frame_shape)
    h, w = frame_shape[:2]
    pts = np.array(BIRD_ROI_POINTS_NORM, dtype=np.float32)
    pts[:, 0] *= max(w - 1, 1)
    pts[:, 1] *= max(h - 1, 1)
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts.astype(np.float32)


def get_birdeye_src_polygon(frame: np.ndarray) -> np.ndarray:
    """
    Returns the 4 source points of the bird-eye quadrilateral on the ORIGINAL frame.

    In this version each point can be dragged with the mouse on the original frame:
    - TL = top-left
    - TR = top-right
    - BR = bottom-right
    - BL = bottom-left

    This changes only the bird-eye projection/visualization. It does not alter
    ALG3 segmentation on the original left panel.
    """
    _ensure_birdeye_roi_points(frame)
    return _roi_points_abs_from_norm(frame.shape)


def make_bird_eye_view(frame: np.ndarray, dst_w: int = 420, dst_h: int = 320) -> np.ndarray:
    src = get_birdeye_src_polygon(frame)
    dst = np.float32([
        [0, 0],
        [dst_w - 1, 0],
        [dst_w - 1, dst_h - 1],
        [0, dst_h - 1],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (dst_w, dst_h))


def draw_birdeye_zone(frame: np.ndarray) -> np.ndarray:
    out = frame.copy()
    src = get_birdeye_src_polygon(frame).astype(np.int32)
    labels = ["TL", "TR", "BR", "BL"]

    cv2.polylines(out, [src.reshape(-1, 1, 2)], True, (0, 255, 255), 3)
    for i, (x, y) in enumerate(src):
        point_color = (0, 0, 255)
        radius = 7
        if BIRD_DRAGGING_POINT_IDX == i:
            point_color = (255, 0, 255)
            radius = 10
        elif BIRD_MOUSE_HOVER_POINT_IDX == i:
            point_color = (0, 165, 255)
            radius = 9
        cv2.circle(out, (int(x), int(y)), radius, point_color, -1)
        cv2.circle(out, (int(x), int(y)), radius + 2, (255, 255, 255), 1)
        cv2.putText(out, labels[i], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, point_color, 2, cv2.LINE_AA)

    cv2.putText(out, "Drag red ROI points with mouse", (int(src[0][0]), max(22, int(src[0][1]) - 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def create_birdeye_roi_trackbar_window(
    *,
    center_x_percent: int = 50,
    top_y_percent: int = 55,
    bottom_y_percent: int = 98,
    top_width_percent: int = 40,
    bottom_width_percent: int = 100,
    lane_width_percent: int = 70,
) -> None:
    """
    Opens a small control window. The ROI itself is controlled with the mouse;
    this window keeps only the Lane Width % slider because that parameter affects
    single-side center estimation.
    """
    global ROI_TRACKBARS_ACTIVE
    cv2.namedWindow(ROI_TRACKBAR_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(ROI_TRACKBAR_WINDOW, 560, 220)

    panel = np.zeros((220, 560, 3), dtype=np.uint8)
    cv2.putText(panel, "Mouse-draggable bird-eye ROI", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)
    cv2.putText(panel, "Drag TL/TR/BR/BL red points on the original video.", (12, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(panel, "SPACE=pause | R=reset ROI | T=show/hide this panel", (12, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (200, 200, 200), 1)
    cv2.putText(panel, "Lane Width % is still adjustable here.", (12, 119), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 255), 1)
    cv2.imshow(ROI_TRACKBAR_WINDOW, panel)

    cv2.createTrackbar("Lane Width %", ROI_TRACKBAR_WINDOW, int(lane_width_percent), 150, lambda _x: None)
    ROI_TRACKBARS_ACTIVE = True


def update_birdeye_roi_from_trackbars() -> None:
    """
    Reads the control-window values. ROI points are not read from sliders anymore;
    they are updated by the mouse callback.
    """
    global ROI_TRACKBARS_ACTIVE, LANE_WIDTH_PERCENT_RUNTIME

    if not ROI_TRACKBARS_ACTIVE:
        return

    try:
        visible = cv2.getWindowProperty(ROI_TRACKBAR_WINDOW, cv2.WND_PROP_VISIBLE)
        if visible < 1:
            ROI_TRACKBARS_ACTIVE = False
            return
        LANE_WIDTH_PERCENT_RUNTIME = max(1, cv2.getTrackbarPos("Lane Width %", ROI_TRACKBAR_WINDOW))
    except cv2.error:
        ROI_TRACKBARS_ACTIVE = False


def toggle_birdeye_roi_trackbar_window() -> None:
    global ROI_TRACKBARS_ACTIVE
    if ROI_TRACKBARS_ACTIVE:
        try:
            cv2.destroyWindow(ROI_TRACKBAR_WINDOW)
        except cv2.error:
            pass
        ROI_TRACKBARS_ACTIVE = False
    else:
        create_birdeye_roi_trackbar_window(lane_width_percent=LANE_WIDTH_PERCENT_RUNTIME)


def draw_runtime_roi_values(img: np.ndarray) -> np.ndarray:
    """Adds current draggable ROI point coordinates to the composed display."""
    out = img.copy()
    if LAST_FRAME_SHAPE is not None and BIRD_ROI_POINTS_NORM is not None:
        pts = _roi_points_abs_from_norm(LAST_FRAME_SHAPE).astype(int)
        txt = (
            f"ROI pts: TL=({pts[0][0]},{pts[0][1]}) TR=({pts[1][0]},{pts[1][1]}) "
            f"BR=({pts[2][0]},{pts[2][1]}) BL=({pts[3][0]},{pts[3][1]}) "
            f"LaneW={LANE_WIDTH_PERCENT_RUNTIME}%"
        )
    else:
        txt = f"ROI pts: not initialized | LaneW={LANE_WIDTH_PERCENT_RUNTIME}%"
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 28), (0, 0, 0), -1)
    cv2.putText(out, txt, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (0, 255, 255), 1, cv2.LINE_AA)
    return out


def on_main_window_mouse(event, x, y, flags, param) -> None:
    """Mouse interaction for moving the 4 bird-eye ROI corners on the left/original panel."""
    global BIRD_DRAGGING_POINT_IDX, BIRD_MOUSE_HOVER_POINT_IDX, BIRD_ROI_POINTS_NORM

    if LAST_FRAME_SHAPE is None:
        return

    h, w = LAST_FRAME_SHAPE[:2]
    scale = MAIN_DISPLAY_SCALE if MAIN_DISPLAY_SCALE > 0 else 1.0
    xu = float(x) / scale
    yu = float(y) / scale

    # Only the left/original panel is draggable. The original frame occupies x=[0,w), y=[0,h).
    if not (0 <= xu < w and 0 <= yu < h):
        if event in (cv2.EVENT_LBUTTONUP, cv2.EVENT_RBUTTONUP):
            BIRD_DRAGGING_POINT_IDX = None
        BIRD_MOUSE_HOVER_POINT_IDX = None
        return

    if BIRD_ROI_POINTS_NORM is None:
        reset_birdeye_roi_points(LAST_FRAME_SHAPE)

    pts = _roi_points_abs_from_norm(LAST_FRAME_SHAPE)
    dists = np.sqrt((pts[:, 0] - xu) ** 2 + (pts[:, 1] - yu) ** 2)
    nearest_idx = int(np.argmin(dists))
    threshold_px = 28.0

    if event == cv2.EVENT_LBUTTONDOWN:
        if dists[nearest_idx] <= threshold_px:
            BIRD_DRAGGING_POINT_IDX = nearest_idx
            BIRD_MOUSE_HOVER_POINT_IDX = nearest_idx

    elif event == cv2.EVENT_MOUSEMOVE:
        if BIRD_DRAGGING_POINT_IDX is not None:
            idx = BIRD_DRAGGING_POINT_IDX
            BIRD_ROI_POINTS_NORM[idx] = [
                float(np.clip(xu / max(w - 1, 1), 0.0, 1.0)),
                float(np.clip(yu / max(h - 1, 1), 0.0, 1.0)),
            ]
            BIRD_ROI_POINTS_NORM = _enforce_roi_point_order(BIRD_ROI_POINTS_NORM)
            BIRD_MOUSE_HOVER_POINT_IDX = idx
        else:
            BIRD_MOUSE_HOVER_POINT_IDX = nearest_idx if dists[nearest_idx] <= threshold_px else None

    elif event == cv2.EVENT_LBUTTONUP:
        BIRD_DRAGGING_POINT_IDX = None
        BIRD_MOUSE_HOVER_POINT_IDX = nearest_idx if dists[nearest_idx] <= threshold_px else None

    elif event == cv2.EVENT_RBUTTONDOWN:
        reset_birdeye_roi_points(LAST_FRAME_SHAPE)


def _split_xs_into_spans(xs: np.ndarray) -> list[tuple[int, int]]:
    if len(xs) == 0:
        return []
    splits = np.where(np.diff(xs) > 1)[0]
    starts = np.r_[0, splits + 1]
    ends = np.r_[splits, len(xs) - 1]
    return [(int(xs[s]), int(xs[e])) for s, e in zip(starts, ends)]


def _choose_span(spans: list[tuple[int, int]], center_x: int) -> tuple[int, int] | None:
    if not spans:
        return None
    for sp in spans:
        if sp[0] <= center_x <= sp[1]:
            return sp
    return max(spans, key=lambda sp: sp[1] - sp[0])


def _fit_x_as_function_of_y(points: list[tuple[float, float]], requested_degree: int):
    """Fit boundary curve x = f(y)."""
    if len(points) < 2:
        return None, 0
    deg = min(max(1, requested_degree), len(points) - 1)
    pts = np.array(points, dtype=np.float32)
    xs = pts[:, 0]
    ys = pts[:, 1]
    coeff = np.polyfit(ys, xs, deg=deg)
    return np.poly1d(coeff), deg


def _x_from_poly(poly, y: float) -> float | None:
    if poly is None:
        return None
    return float(poly(y))


def _draw_circle_if_visible(img, x, y, color, radius=4):
    h, w = img.shape[:2]
    xi, yi = int(round(x)), int(round(y))
    if 0 <= xi < w and 0 <= yi < h:
        cv2.circle(img, (xi, yi), radius, color, -1)


def _draw_polyline_clipped(img: np.ndarray, pts: list[tuple[float, float]], color, thickness=2):
    if len(pts) < 2:
        return
    arr = np.array([[int(round(x)), int(round(y))] for x, y in pts], dtype=np.int32)
    cv2.polylines(img, [arr.reshape(-1, 1, 2)], False, color, thickness)


def alg3_dark_road_midline(
    frame: np.ndarray,
    *,
    title: str = "ALG3 dark-road midline",
    roi_y_ratio: float = 0.45,
    draw_birdeye_roi: bool = False,
    create_bird_eye: bool = False,
    bird_dst_w: int = 420,
    bird_dst_h: int = 320,
    use_single_side_estimation: bool = True,
    lane_width_ratio: float = 0.70,
    lane_width_px: float | None = None,
    edge_margin_ratio: float = 0.035,
    min_visible_width_ratio: float = 0.55,
    min_boundary_points: int = 3,
    curve_fit_degree: int = 2,
    row_count: int = 14,
) -> Result:
    """
    ALG3 with a centro_giallo-style single-side fallback.

    Unchanged ALG3 part:
    - lower ROI;
    - HSV dark-road segmentation;
    - largest connected component;
    - horizontal span extraction.

    Modified midline part:
    - when both sides are visible, center = average of left and right boundary;
    - when only the left boundary is reliable, center = left + lane_width/2;
    - when only the right boundary is reliable, center = right - lane_width/2.

    This mirrors the idea used by the working lane code: one line is enough if
    the lane width is known/estimated.
    """
    h, w = frame.shape[:2]
    roi, y0 = roi_frame(frame, roi_y_ratio)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    _, S, V = cv2.split(hsv)

    # Original ALG3 dark-road segmentation.
    mask_v = cv2.inRange(V, 0, 110)
    mask_s = cv2.inRange(S, 0, 120)
    road = cv2.bitwise_and(mask_v, mask_s)

    kernel = np.ones((7, 7), np.uint8)
    road = cv2.morphologyEx(road, cv2.MORPH_OPEN, kernel)
    road = cv2.morphologyEx(road, cv2.MORPH_CLOSE, kernel)

    comp, stats, _ = largest_component(road)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y0), (w - 1, h - 1), (80, 80, 80), 1)

    def finalize(ok: bool, error: float | None, overlay_img: np.ndarray, used: bool = False, mode: str = "none") -> Result:
        if draw_birdeye_roi:
            overlay_img = draw_birdeye_zone(overlay_img)
        bird_eye = make_bird_eye_view(frame, bird_dst_w, bird_dst_h) if create_bird_eye else None
        return Result(ok, error, overlay_img, bird_eye, used, mode)

    if comp is None or stats[cv2.CC_STAT_AREA] < 500:
        overlay = put_text(overlay, [title, "no road component"])
        return finalize(False, None, overlay)

    road = (comp.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(road, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay[y0:, :], contours, -1, (0, 255, 0), 2)

    # More rows than the original ALG3 help the line/curve fit in bends.
    ys = np.linspace(int(roi.shape[0] * 0.20), roi.shape[0] - 8, row_count).astype(int)
    edge_margin_px = max(4, int(w * edge_margin_ratio))
    min_span_width = 40

    row_samples: list[dict] = []
    for yy in ys:
        row = road[yy, :]
        xs = np.where(row > 0)[0]
        if len(xs) < 20:
            continue

        spans = _split_xs_into_spans(xs)
        spans = [sp for sp in spans if (sp[1] - sp[0]) > min_span_width]
        chosen = _choose_span(spans, w // 2)
        if chosen is None:
            continue

        xl, xr = chosen
        width = xr - xl
        row_samples.append({
            "yy": int(yy),
            "xl": int(xl),
            "xr": int(xr),
            "width": int(width),
            "left_clipped": bool(xl <= edge_margin_px),
            "right_clipped": bool(xr >= (w - 1 - edge_margin_px)),
        })

    if len(row_samples) < 3:
        overlay = put_text(overlay, [title, "insufficient spans"])
        return finalize(False, None, overlay)

    lane_width = float(lane_width_px) if lane_width_px is not None else float(w * lane_width_ratio)
    lane_width = float(np.clip(lane_width, 40.0, w * 1.50))

    # Build left/right boundary point sets from the ALG3 span endpoints.
    # Clipped boundaries are not used to fit a side line, because they are often
    # just the image/ROI border and not the real lane boundary.
    left_points: list[tuple[float, float]] = []
    right_points: list[tuple[float, float]] = []
    for s in row_samples:
        yy = float(s["yy"])
        if not s["left_clipped"]:
            left_points.append((float(s["xl"]), yy))
        if not s["right_clipped"]:
            right_points.append((float(s["xr"]), yy))

    left_poly, left_deg = _fit_x_as_function_of_y(left_points, curve_fit_degree) if len(left_points) >= min_boundary_points else (None, 0)
    right_poly, right_deg = _fit_x_as_function_of_y(right_points, curve_fit_degree) if len(right_points) >= min_boundary_points else (None, 0)

    mids: list[tuple[float, float]] = []
    estimated_hidden_edges: list[tuple[float, float]] = []
    estimated_centers: list[tuple[float, float]] = []
    used_single_side = False
    used_modes: set[str] = set()

    for s in row_samples:
        yy = float(s["yy"])
        xl = float(s["xl"])
        xr = float(s["xr"])
        span_center = 0.5 * (xl + xr)
        normal_mid = span_center

        left_x = _x_from_poly(left_poly, yy)
        right_x = _x_from_poly(right_poly, yy)

        # A side is reliable on this row only if it was not clipped. Otherwise
        # that endpoint is probably the image border, not the real curve.
        row_has_left = left_x is not None and not s["left_clipped"]
        row_has_right = right_x is not None and not s["right_clipped"]

        # If the visible span is much narrower than the expected lane width, treat
        # it as a partial view even if it does not exactly touch the border.
        narrow = s["width"] < lane_width * min_visible_width_ratio
        likely_only_right = narrow and span_center >= (w / 2.0) and right_x is not None
        likely_only_left = narrow and span_center < (w / 2.0) and left_x is not None

        mode = "normal"
        hidden_x = None
        center_x = normal_mid

        if use_single_side_estimation:
            if row_has_left and row_has_right and not narrow:
                center_x = 0.5 * (left_x + right_x)
                mode = "both"
            elif (row_has_right and not row_has_left) or likely_only_right or (right_x is not None and left_poly is None):
                # Right boundary visible, left boundary missing. Estimate center
                # by shifting left by half lane width, exactly like centro_giallo
                # estimates from one detected line.
                center_x = right_x - lane_width / 2.0
                hidden_x = right_x - lane_width
                mode = "left-from-right"
                used_single_side = True
                used_modes.add(mode)
            elif (row_has_left and not row_has_right) or likely_only_left or (left_x is not None and right_poly is None):
                # Left boundary visible, right boundary missing.
                center_x = left_x + lane_width / 2.0
                hidden_x = left_x + lane_width
                mode = "right-from-left"
                used_single_side = True
                used_modes.add(mode)
            elif left_x is not None and right_x is not None:
                center_x = 0.5 * (left_x + right_x)
                mode = "both-fit"
            else:
                center_x = normal_mid
                mode = "normal-fallback"
        else:
            center_x = normal_mid
            mode = "normal-disabled"

        mids.append((center_x, yy))

        # Visual debug.
        _draw_circle_if_visible(overlay, xl, y0 + yy, (255, 0, 0), radius=3)   # detected left endpoint
        _draw_circle_if_visible(overlay, xr, y0 + yy, (255, 0, 0), radius=3)   # detected right endpoint

        if mode in {"left-from-right", "right-from-left"}:
            estimated_centers.append((center_x, yy))
            if hidden_x is not None:
                estimated_hidden_edges.append((hidden_x, yy))
            _draw_circle_if_visible(overlay, center_x, y0 + yy, (255, 0, 255), radius=5)
            if hidden_x is not None:
                _draw_circle_if_visible(overlay, hidden_x, y0 + yy, (255, 0, 255), radius=3)
        else:
            _draw_circle_if_visible(overlay, center_x, y0 + yy, (0, 0, 255), radius=3)

    if len(mids) < 3:
        overlay = put_text(overlay, [title, "insufficient midpoints"])
        return finalize(False, None, overlay, used_single_side)

    # Final midline used by the algorithm.
    center_pts = np.array([[int(round(x)), int(round(y))] for x, y in mids], dtype=np.int32)
    cv2.polylines(overlay[y0:, :], [center_pts.reshape(-1, 1, 2)], False, (0, 255, 255), 2)

    # Draw estimated hidden side and estimated center samples in magenta.
    _draw_polyline_clipped(overlay[y0:, :], estimated_hidden_edges, (255, 0, 255), thickness=1)
    _draw_polyline_clipped(overlay[y0:, :], estimated_centers, (255, 0, 255), thickness=2)

    target_x = float(np.mean(center_pts[-3:, 0]))
    error = target_x - (w // 2)

    cv2.line(overlay, (w // 2, h - 1), (w // 2, y0), (255, 255, 0), 2)
    cv2.line(overlay, (w // 2, h - 30), (int(round(target_x)), h - 35), (0, 0, 255), 2)
    _draw_circle_if_visible(overlay, target_x, h - 35, (0, 0, 255), radius=7)

    mode_txt = "+".join(sorted(used_modes)) if used_modes else "none"
    overlay = put_text(
        overlay,
        [
            title,
            f"error_px={error:+.1f}",
            f"single-side={int(used_single_side)} mode={mode_txt}",
            f"lane_width={lane_width:.0f}px Lpts={len(left_points)} Rpts={len(right_points)}",
        ],
    )

    return finalize(True, float(error), overlay, used_single_side, mode_txt)


def add_title(img: np.ndarray, title: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1] - 1, 34), (0, 0, 0), -1)
    cv2.putText(out, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def format_error(error_px: float | None) -> str:
    return f"{error_px:+.1f}" if error_px is not None else "NA"


def compose_side_by_side(
    original_overlay: np.ndarray,
    bird_eye_alg3_overlay: np.ndarray,
    original_error_px: float | None,
    bird_error_px: float | None,
    original_used_single_side: bool,
    bird_used_single_side: bool,
    right_scale: float = 0.85,
) -> np.ndarray:
    left_panel = add_title(original_overlay, "Original: ALG3 + single-side center")
    H, W = left_panel.shape[:2]

    target_h = int(H * right_scale)
    scale = target_h / bird_eye_alg3_overlay.shape[0]
    target_w = int(bird_eye_alg3_overlay.shape[1] * scale)

    bird_big = cv2.resize(bird_eye_alg3_overlay, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    bird_panel = add_title(bird_big, "Bird-eye: ALG3 + single-side center")

    gap = 24
    info_h = 112
    right_w = bird_panel.shape[1]
    right_h = bird_panel.shape[0] + info_h

    canvas_w = W + gap + right_w
    canvas_h = max(H, right_h)
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
    canvas[:H, :W] = left_panel

    x_right = W + gap
    y_bird = max(0, (canvas_h - right_h) // 2)
    canvas[y_bird:y_bird + bird_panel.shape[0], x_right:x_right + right_w] = bird_panel

    info = np.zeros((info_h, right_w, 3), dtype=np.uint8)
    cv2.putText(info, f"orig error_px={format_error(original_error_px)}", (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(info, f"bird error_px={format_error(bird_error_px)}", (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(info, f"single-side orig={int(original_used_single_side)} bird={int(bird_used_single_side)}", (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(info, "Magenta = estimated missing side/center", (10, 106), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180, 180, 180), 1, cv2.LINE_AA)

    y_info = y_bird + bird_panel.shape[0]
    canvas[y_info:y_info + info_h, x_right:x_right + right_w] = info
    return canvas


def process_frame_realtime(
    frame: np.ndarray,
    *,
    right_scale: float,
    bird_dst_w: int,
    bird_dst_h: int,
    bird_roi_y_ratio: float,
    use_single_side_estimation: bool,
    lane_width_ratio: float,
    lane_width_px: float | None,
    edge_margin_ratio: float,
    min_visible_width_ratio: float,
    min_boundary_points: int,
    curve_fit_degree: int,
    row_count: int,
):
    original_res = alg3_dark_road_midline(
        frame,
        title="ALG3 original",
        roi_y_ratio=0.45,
        draw_birdeye_roi=True,
        create_bird_eye=True,
        bird_dst_w=bird_dst_w,
        bird_dst_h=bird_dst_h,
        use_single_side_estimation=use_single_side_estimation,
        lane_width_ratio=lane_width_ratio,
        lane_width_px=lane_width_px,
        edge_margin_ratio=edge_margin_ratio,
        min_visible_width_ratio=min_visible_width_ratio,
        min_boundary_points=min_boundary_points,
        curve_fit_degree=curve_fit_degree,
        row_count=row_count,
    )

    bird_res = alg3_dark_road_midline(
        original_res.bird_eye,
        title="ALG3 on bird-eye",
        roi_y_ratio=bird_roi_y_ratio,
        draw_birdeye_roi=False,
        create_bird_eye=False,
        use_single_side_estimation=use_single_side_estimation,
        lane_width_ratio=lane_width_ratio,
        lane_width_px=lane_width_px,
        edge_margin_ratio=edge_margin_ratio,
        min_visible_width_ratio=min_visible_width_ratio,
        min_boundary_points=min_boundary_points,
        curve_fit_degree=curve_fit_degree,
        row_count=row_count,
    )

    composed = compose_side_by_side(
        original_res.overlay,
        bird_res.overlay,
        original_res.error_px,
        bird_res.error_px,
        original_res.used_single_side_estimation,
        bird_res.used_single_side_estimation,
        right_scale=right_scale,
    )
    return composed, original_res, bird_res


def draw_video_realtime_status(img: np.ndarray, fps_processing: float, frame_idx: int, video_fps: float, playback_time_s: float, paused: bool) -> np.ndarray:
    out = img.copy()
    state = "PAUSED" if paused else "PLAY"
    status = (
        f"VIDEO REALTIME | {state} | frame={frame_idx} | t={playback_time_s:.2f}s | "
        f"video_fps={video_fps:.1f} | proc_fps={fps_processing:.1f} | "
        f"Q/ESC quit | SPACE pause | drag ROI points | R reset | T controls"
    )
    cv2.rectangle(out, (0, out.shape[0] - 34), (out.shape[1] - 1, out.shape[0] - 1), (0, 0, 0), -1)
    cv2.putText(out, status, (10, out.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def run_video_realtime(
    video_path: str,
    *,
    right_scale: float = 0.85,
    bird_dst_w: int = 420,
    bird_dst_h: int = 320,
    bird_roi_y_ratio: float = 0.0,
    display_scale: float = 1.0,
    print_errors: bool = False,
    loop: bool = False,
    sync_to_video_fps: bool = True,
    use_single_side_estimation: bool = True,
    lane_width_ratio: float = 0.70,
    lane_width_px: float | None = None,
    edge_margin_ratio: float = 0.035,
    min_visible_width_ratio: float = 0.55,
    min_boundary_points: int = 3,
    curve_fit_degree: int = 2,
    row_count: int = 14,
    enable_roi_trackbars: bool = True,
    bird_center_x_percent: int = 50,
    bird_top_y_percent: int = 55,
    bird_bottom_y_percent: int = 98,
    bird_top_width_percent: int = 40,
    bird_bottom_width_percent: int = 100,
):
    global BIRD_CENTER_X_PERCENT, BIRD_TOP_Y_PERCENT, BIRD_BOTTOM_Y_PERCENT
    global BIRD_TOP_WIDTH_PERCENT, BIRD_BOTTOM_WIDTH_PERCENT, LANE_WIDTH_PERCENT_RUNTIME
    global BIRD_ROI_POINTS_NORM, LAST_FRAME_SHAPE, MAIN_DISPLAY_SCALE

    # Initialize the runtime ROI parameters from the command-line defaults.
    BIRD_CENTER_X_PERCENT = int(bird_center_x_percent)
    BIRD_TOP_Y_PERCENT = int(bird_top_y_percent)
    BIRD_BOTTOM_Y_PERCENT = int(bird_bottom_y_percent)
    BIRD_TOP_WIDTH_PERCENT = int(bird_top_width_percent)
    BIRD_BOTTOM_WIDTH_PERCENT = int(bird_bottom_width_percent)
    LANE_WIDTH_PERCENT_RUNTIME = int(round(lane_width_ratio * 100))
    BIRD_ROI_POINTS_NORM = None
    MAIN_DISPLAY_SCALE = float(display_scale)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_period = 1.0 / max(video_fps, 1e-6)
    cv2.namedWindow(MAIN_WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(MAIN_WINDOW_NAME, on_main_window_mouse)

    if enable_roi_trackbars:
        create_birdeye_roi_trackbar_window(
            center_x_percent=BIRD_CENTER_X_PERCENT,
            top_y_percent=BIRD_TOP_Y_PERCENT,
            bottom_y_percent=BIRD_BOTTOM_Y_PERCENT,
            top_width_percent=BIRD_TOP_WIDTH_PERCENT,
            bottom_width_percent=BIRD_BOTTOM_WIDTH_PERCENT,
            lane_width_percent=LANE_WIDTH_PERCENT_RUNTIME,
        )

    frame_idx = 0
    fps_smooth = 0.0
    paused = False
    last_display = None
    last_playback_time_s = 0.0

    try:
        last_raw_frame = None

        while True:
            loop_start = time.perf_counter()

            if enable_roi_trackbars:
                update_birdeye_roi_from_trackbars()

            # When paused, reprocess the last frame so that ROI slider changes
            # are visible immediately without advancing the video.
            if paused:
                if last_raw_frame is None:
                    frame_to_process = None
                    display_frame_idx = max(frame_idx - 1, 0)
                else:
                    frame_to_process = last_raw_frame
                    display_frame_idx = max(frame_idx - 1, 0)
            else:
                ret, frame = cap.read()
                if not ret:
                    if loop:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        frame_idx = 0
                        last_raw_frame = None
                        continue
                    print("End of video. Stopping realtime playback.")
                    break

                last_raw_frame = frame
                frame_to_process = frame
                display_frame_idx = frame_idx

            if frame_to_process is not None:
                LAST_FRAME_SHAPE = frame_to_process.shape
                _ensure_birdeye_roi_points(frame_to_process)

                effective_lane_width_ratio = (
                    LANE_WIDTH_PERCENT_RUNTIME / 100.0
                    if enable_roi_trackbars
                    else lane_width_ratio
                )

                composed, original_res, bird_res = process_frame_realtime(
                    frame_to_process,
                    right_scale=right_scale,
                    bird_dst_w=bird_dst_w,
                    bird_dst_h=bird_dst_h,
                    bird_roi_y_ratio=bird_roi_y_ratio,
                    use_single_side_estimation=use_single_side_estimation,
                    lane_width_ratio=effective_lane_width_ratio,
                    lane_width_px=lane_width_px,
                    edge_margin_ratio=edge_margin_ratio,
                    min_visible_width_ratio=min_visible_width_ratio,
                    min_boundary_points=min_boundary_points,
                    curve_fit_degree=curve_fit_degree,
                    row_count=row_count,
                )

                process_end = time.perf_counter()
                proc_dt = max(process_end - loop_start, 1e-6)
                inst_fps = 1.0 / proc_dt
                fps_smooth = inst_fps if display_frame_idx == 0 else (0.9 * fps_smooth + 0.1 * inst_fps)

                last_playback_time_s = display_frame_idx / video_fps

                if enable_roi_trackbars:
                    composed = draw_runtime_roi_values(composed)

                composed = draw_video_realtime_status(
                    composed,
                    fps_processing=fps_smooth,
                    frame_idx=display_frame_idx,
                    video_fps=video_fps,
                    playback_time_s=last_playback_time_s,
                    paused=paused,
                )

                if display_scale != 1.0:
                    disp_w = int(composed.shape[1] * display_scale)
                    disp_h = int(composed.shape[0] * display_scale)
                    composed_display = cv2.resize(composed, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
                else:
                    composed_display = composed

                last_display = composed_display

                if print_errors and (not paused) and frame_idx % 10 == 0:
                    print(
                        f"frame={frame_idx:06d} | "
                        f"orig_ok={int(original_res.ok)} orig_error={format_error(original_res.error_px)} "
                        f"orig_single={int(original_res.used_single_side_estimation)} mode={original_res.estimation_mode} | "
                        f"bird_ok={int(bird_res.ok)} bird_error={format_error(bird_res.error_px)} "
                        f"bird_single={int(bird_res.used_single_side_estimation)} mode={bird_res.estimation_mode} | "
                        f"proc_fps={fps_smooth:.1f} | "
                        f"ROI_pts={_roi_points_abs_from_norm(LAST_FRAME_SHAPE).astype(int).tolist()}"
                    )

                if not paused:
                    frame_idx += 1

            if last_display is not None:
                cv2.imshow(MAIN_WINDOW_NAME, last_display)

            if paused or not sync_to_video_fps:
                wait_ms = 1
            else:
                elapsed = time.perf_counter() - loop_start
                remaining = frame_period - elapsed
                wait_ms = max(1, int(remaining * 1000))

            key = cv2.waitKey(wait_ms) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key == ord(" "):
                paused = not paused
            if key in (ord("r"), ord("R")) and LAST_FRAME_SHAPE is not None:
                reset_birdeye_roi_points(LAST_FRAME_SHAPE)
            if key in (ord("t"), ord("T")) and enable_roi_trackbars:
                toggle_birdeye_roi_trackbar_window()

    finally:
        cap.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Realtime display from a VIDEO FILE: ALG3 + centro_giallo-style single-side centerline + bird-eye diagnostic view. No output video is saved."
    )
    parser.add_argument("--video", required=True, help="Input video path, e.g. test3.mp4")
    parser.add_argument("--right-scale", type=float, default=0.85, help="Bird-eye panel height as fraction of original height")
    parser.add_argument("--bird-dst-w", type=int, default=420, help="Bird-eye warped image width before display scaling")
    parser.add_argument("--bird-dst-h", type=int, default=320, help="Bird-eye warped image height before display scaling")
    parser.add_argument("--bird-roi-y-ratio", type=float, default=0.0, help="ROI y-ratio used when applying ALG3 on the bird-eye view")
    parser.add_argument("--display-scale", type=float, default=1.0, help="Scale only the displayed composed window")
    parser.add_argument("--print-errors", action="store_true", help="Print original/bird-eye error_px values every 10 frames")
    parser.add_argument("--loop", action="store_true", help="Restart from the first frame when the video ends")
    parser.add_argument("--no-sync", action="store_true", help="Do not wait for the original video FPS; process/display as fast as possible")
    parser.add_argument("--disable-single-side-estimation", action="store_true", help="Disable the single-side lane-width estimator and use the old span midpoint")
    parser.add_argument("--lane-width-ratio", type=float, default=0.70, help="Expected lane/road width as fraction of the processed image width")
    parser.add_argument("--lane-width-px", type=float, default=None, help="Optional fixed lane/road width in pixels. Overrides --lane-width-ratio")
    parser.add_argument("--edge-margin-ratio", type=float, default=0.035, help="How close a detected endpoint must be to the image border to be considered clipped")
    parser.add_argument("--min-visible-width-ratio", type=float, default=0.55, help="If visible span width is below lane_width*this, treat it as partial/single-side")
    parser.add_argument("--min-boundary-points", type=int, default=3, help="Minimum points needed to fit a left/right boundary curve")
    parser.add_argument("--curve-fit-degree", type=int, default=2, help="Polynomial degree for boundary curve x=f(y). Use 1 or 2")
    parser.add_argument("--row-count", type=int, default=14, help="Number of horizontal rows sampled inside the ROI")

    # Runtime bird-eye ROI defaults. After startup, drag the four ROI points directly with the mouse.
    parser.add_argument("--no-roi-trackbars", action="store_true", help="Disable the realtime lane-width control window. Mouse ROI dragging still works")
    parser.add_argument("--bird-center-x-percent", type=int, default=50, help="Initial bird-eye trapezoid horizontal center, in percent of frame width")
    parser.add_argument("--bird-top-y-percent", type=int, default=55, help="Initial bird-eye trapezoid top edge Y, in percent of frame height")
    parser.add_argument("--bird-bottom-y-percent", type=int, default=98, help="Initial bird-eye trapezoid bottom edge Y, in percent of frame height")
    parser.add_argument("--bird-top-width-percent", type=int, default=40, help="Initial bird-eye trapezoid top width, in percent of frame width")
    parser.add_argument("--bird-bottom-width-percent", type=int, default=100, help="Initial bird-eye trapezoid bottom width, in percent of frame width")
    args = parser.parse_args()

    run_video_realtime(
        args.video,
        right_scale=args.right_scale,
        bird_dst_w=args.bird_dst_w,
        bird_dst_h=args.bird_dst_h,
        bird_roi_y_ratio=args.bird_roi_y_ratio,
        display_scale=args.display_scale,
        print_errors=args.print_errors,
        loop=args.loop,
        sync_to_video_fps=not args.no_sync,
        use_single_side_estimation=not args.disable_single_side_estimation,
        lane_width_ratio=args.lane_width_ratio,
        lane_width_px=args.lane_width_px,
        edge_margin_ratio=args.edge_margin_ratio,
        min_visible_width_ratio=args.min_visible_width_ratio,
        min_boundary_points=args.min_boundary_points,
        curve_fit_degree=args.curve_fit_degree,
        row_count=args.row_count,
        enable_roi_trackbars=not args.no_roi_trackbars,
        bird_center_x_percent=args.bird_center_x_percent,
        bird_top_y_percent=args.bird_top_y_percent,
        bird_bottom_y_percent=args.bird_bottom_y_percent,
        bird_top_width_percent=args.bird_top_width_percent,
        bird_bottom_width_percent=args.bird_bottom_width_percent,
    )


if __name__ == "__main__":
    main()
