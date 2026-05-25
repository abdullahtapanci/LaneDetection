from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class LaneDepartureState:
    is_warning: bool
    direction: str | None
    offset_ratio: float | None
    message: str


@dataclass(frozen=True)
class EgoLaneSelection:
    left_idx: int | None
    right_idx: int | None
    left_x: float | None
    right_x: float | None
    lane_width: float | None
    lane_center_x: float | None
    score: float | None


DEFAULT_EGO_REFERENCE_POINT = (0.5, 0.9)


def get_ego_reference_point(image_width, image_height):
    x_ratio, y_ratio = DEFAULT_EGO_REFERENCE_POINT
    return float(x_ratio * image_width), int(y_ratio * image_height)


def select_ego_lanes(
    lanes,
    image_width,
    image_height,
    reference_point=None,
):
    if len(lanes) < 2:
        return EgoLaneSelection(None, None, None, None, None, None, None)

    if reference_point is None:
        reference_x, reference_y = get_ego_reference_point(image_width, image_height)
    else:
        reference_x, reference_y = reference_point
    reference_y = int(np.clip(reference_y, 0, image_height - 1))

    left_candidates = []
    right_candidates = []
    for idx, lane in enumerate(lanes):
        if not (lane["y_min"] <= reference_y <= lane["y_max"]):
            continue

        lane_x = float(np.polyval(lane["poly"], reference_y))
        distance = abs(reference_x - lane_x)
        if lane_x < reference_x:
            left_candidates.append((distance, idx, lane_x))
        elif lane_x > reference_x:
            right_candidates.append((distance, idx, lane_x))

    if not left_candidates or not right_candidates:
        return EgoLaneSelection(None, None, None, None, None, None, None)

    left_distance, left_idx, left_x = min(left_candidates, key=lambda item: item[0])
    right_distance, right_idx, right_x = min(right_candidates, key=lambda item: item[0])
    lane_width = right_x - left_x
    lane_center_x = (left_x + right_x) / 2.0
    score = (left_distance + right_distance) / image_width

    return EgoLaneSelection(
        left_idx,
        right_idx,
        left_x,
        right_x,
        lane_width,
        lane_center_x,
        score,
    )


def assess_lane_departure(
    lanes,
    left_idx,
    right_idx,
    image_width,
    image_height,
    warning_threshold=0.28,
    evaluation_y_ratio=0.9,
):
    if left_idx is None or right_idx is None:
        return LaneDepartureState(False, None, None, "Ego lanes unavailable")

    left_lane = lanes[left_idx]
    right_lane = lanes[right_idx]
    y_eval = int(image_height * evaluation_y_ratio)
    y_eval = max(left_lane["y_min"], right_lane["y_min"], min(y_eval, left_lane["y_max"], right_lane["y_max"]))

    left_x = float(np.polyval(left_lane["poly"], y_eval))
    right_x = float(np.polyval(right_lane["poly"], y_eval))
    if right_x <= left_x:
        return LaneDepartureState(False, None, None, "Invalid ego lane geometry")

    vehicle_center_x = image_width / 2.0
    lane_center_x = (left_x + right_x) / 2.0
    lane_width = right_x - left_x
    offset_ratio = (vehicle_center_x - lane_center_x) / lane_width

    if abs(offset_ratio) < warning_threshold:
        return LaneDepartureState(False, None, offset_ratio, "Centered")

    direction = "right" if offset_ratio > 0 else "left"
    return LaneDepartureState(
        True,
        direction,
        offset_ratio,
        f"Lane departure warning: drifting {direction}",
    )


def draw_lane_departure_warning(image_rgb, lanes, left_idx, right_idx, state, alpha=0.65):
    if not state.is_warning or left_idx is None or right_idx is None:
        return image_rgb

    height, width = image_rgb.shape[:2]
    left_lane = lanes[left_idx]
    right_lane = lanes[right_idx]
    y_min = max(left_lane["y_min"], right_lane["y_min"])
    y_max = min(left_lane["y_max"], right_lane["y_max"])
    if y_min >= y_max:
        return image_rgb

    ys = np.arange(y_min, y_max + 1, 2)
    xs_left = np.clip(np.polyval(left_lane["poly"], ys).astype(int), 0, width - 1)
    xs_right = np.clip(np.polyval(right_lane["poly"], ys).astype(int), 0, width - 1)
    pts = np.concatenate([
        np.stack([xs_left, ys], axis=1),
        np.stack([xs_right, ys], axis=1)[::-1],
    ]).reshape(-1, 1, 2)

    overlay = image_rgb.copy()
    cv2.fillPoly(overlay, [pts], (255, 220, 0))
    warned = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

    cv2.putText(
        warned,
        "LANE DEPARTURE",
        (24, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (255, 190, 0),
        3,
        cv2.LINE_AA,
    )
    return warned
