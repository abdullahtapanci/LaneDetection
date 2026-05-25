import cv2
import numpy as np


FRAME_SIZE = (640, 480)
DEFAULT_LOWER_HSV = (0, 0, 200)
DEFAULT_UPPER_HSV = (179, 50, 255)

PERSPECTIVE_PRESETS = {
    "testVideo1": {
        "top_left": (240, 250),
        "bottom_left": (60, 472),
        "top_right": (340, 250),
        "bottom_right": (520, 472),
    },
    "testVideo2": {
        "top_left": (248, 280),
        "bottom_left": (168, 420),
        "top_right": (398, 280),
        "bottom_right": (450, 420),
    },
    "testVideo3": {
        "top_left": (220, 350),
        "bottom_left": (140, 440),
        "top_right": (320, 350),
        "bottom_right": (400, 440),
    },
    "testVideo4": {
        "top_left": (310, 350),
        "bottom_left": (230, 450),
        "top_right": (360, 350),
        "bottom_right": (420, 450),
    },
    "testVideo5": {
        "top_left": (250, 400),
        "bottom_left": (220, 470),
        "top_right": (380, 400),
        "bottom_right": (410, 470),
    },
}

MEDIA_PERSPECTIVE_PRESETS = {
    "testVideo1.mp4": "testVideo1",
    "testVideo2.mp4": "testVideo2",
    "testVideo3.mov": "testVideo3",
    "testVideo4.mov": "testVideo4",
    "testVideo5.mov": "testVideo5",
}

DEFAULT_PERSPECTIVE_PRESET = "testVideo2"

MEDIA_HSV_PRESETS = {
    "testVideo1.mp4": {
        "lower": (0, 0, 150),
        "upper": (179, 150, 255),
    },
    "testVideo2.mp4": {
        "lower": (0, 0, 150),
        "upper": (179, 150, 255),
    },
    "testVideo3.mov": {
        "lower": (0, 0, 60), #_,_,down
        "upper": (179, 150, 255), #_,up,_
    },
    "testVideo4.mov": {
        "lower": (0, 0, 160), #_,_,down
        "upper": (179, 110, 255), #_,up,_
    },
    "testVideo5.mov": {
        "lower": (0, 0, 160), #_,_,down
        "upper": (179, 110, 255), #_,up,_
    },
}


def _preset_points(preset_name):
    preset = PERSPECTIVE_PRESETS[preset_name]
    return np.float32([
        preset["top_left"],
        preset["bottom_left"],
        preset["top_right"],
        preset["bottom_right"],
    ])


def get_perspective_preset_for_media(media_name):
    return MEDIA_PERSPECTIVE_PRESETS.get(media_name, DEFAULT_PERSPECTIVE_PRESET)


def get_hsv_preset_for_media(media_name):
    return MEDIA_HSV_PRESETS.get(
        media_name,
        {
            "lower": DEFAULT_LOWER_HSV,
            "upper": DEFAULT_UPPER_HSV,
        },
    )


def process_frame(
    frame_bgr,
    lower_hsv=DEFAULT_LOWER_HSV,
    upper_hsv=DEFAULT_UPPER_HSV,
    perspective_preset=None,
    media_name=None,
    window_half_width=50,
    window_height=40,
):
    frame = cv2.resize(frame_bgr, FRAME_SIZE)
    width, height = FRAME_SIZE
    if perspective_preset is None:
        perspective_preset = get_perspective_preset_for_media(media_name)
    if lower_hsv == DEFAULT_LOWER_HSV and upper_hsv == DEFAULT_UPPER_HSV:
        hsv_preset = get_hsv_preset_for_media(media_name)
        lower_hsv = hsv_preset["lower"]
        upper_hsv = hsv_preset["upper"]

    pts1 = _preset_points(perspective_preset)
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, FRAME_SIZE)

    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_transformed_frame,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )

    histogram = np.sum(mask[mask.shape[0] // 2 :, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    y = height - 8
    lx, ly = [], []
    rx, ry = [], []
    sliding_windows = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    while y > 0:
        top = max(0, y - window_height)

        left_start = max(0, left_base - window_half_width)
        left_end = min(width, left_base + window_half_width)
        left_window = mask[top:y, left_start:left_end]
        contours, _ = cv2.findContours(left_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                left_base = left_start + cx
                lx.append(left_base)
                ly.append(top + cy)

        right_start = max(0, right_base - window_half_width)
        right_end = min(width, right_base + window_half_width)
        right_window = mask[top:y, right_start:right_end]
        contours, _ = cv2.findContours(right_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                right_base = right_start + cx
                rx.append(right_base)
                ry.append(top + cy)

        cv2.rectangle(
            sliding_windows,
            (max(0, left_base - window_half_width), y),
            (min(width, left_base + window_half_width), top),
            (255, 0, 0),
            2,
        )
        cv2.rectangle(
            sliding_windows,
            (max(0, right_base - window_half_width), y),
            (min(width, right_base + window_half_width), top),
            (255, 0, 0),
            2,
        )
        y -= window_height

    annotated = frame.copy()
    if len(lx) >= 4 and len(rx) >= 4:
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)
        overlay = np.zeros_like(transformed_frame)

        left_fit = np.polyfit(ly, lx, 1)
        right_fit = np.polyfit(ry, rx, 1)

        y_range = np.arange(0, height)
        left_x = np.clip(np.polyval(left_fit, y_range).astype(int), 0, width - 1)
        right_x = np.clip(np.polyval(right_fit, y_range).astype(int), 0, width - 1)

        pts_left = np.array([left_x, y_range]).T
        pts_right = np.array([right_x, y_range]).T
        fill_poly = np.vstack([pts_left, pts_right[::-1]]).astype(np.int32)
        cv2.fillPoly(overlay, [fill_poly], (0, 200, 0))

        left_draw = pts_left.reshape(-1, 1, 2).astype(np.int32)
        right_draw = pts_right.reshape(-1, 1, 2).astype(np.int32)
        cv2.polylines(overlay, [left_draw], False, (0, 0, 255), 5)
        cv2.polylines(overlay, [right_draw], False, (255, 0, 0), 5)

        inv_overlay = cv2.warpPerspective(overlay, inv_matrix, FRAME_SIZE)
        annotated = cv2.addWeighted(annotated, 1.0, inv_overlay, 0.5, 0)

    for point in pts1.astype(int):
        cv2.circle(annotated, tuple(point), 5, (0, 0, 255), -1)

    return {
        "annotated": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
        "bird_eye": cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGB),
        "mask": cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB),
        "sliding_windows": cv2.cvtColor(sliding_windows, cv2.COLOR_BGR2RGB),
        "detected_points": {
            "left": len(lx),
            "right": len(rx),
        },
    }
