import cv2
import numpy as np


FRAME_SIZE = (640, 480)
DEFAULT_LOWER_HSV = (0, 0, 200)
DEFAULT_UPPER_HSV = (179, 50, 255)

# These are some predefined perspective presets for the test videos.
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

# Here we map the media file names to their corresponding perspective presets.
MEDIA_PERSPECTIVE_PRESETS = {
    "testVideo1.mp4": "testVideo1",
    "testVideo2.mp4": "testVideo2",
    "testVideo3.mov": "testVideo3",
    "testVideo4.mov": "testVideo4",
    "testVideo5.mov": "testVideo5",
}

DEFAULT_PERSPECTIVE_PRESET = "testVideo2"

# These are predefined HSV thresholds for the test videos.
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


# This function takes preset name and return the values as a numpy array.
def _preset_points(preset_name):
    preset = PERSPECTIVE_PRESETS[preset_name]
    return np.float32([
        preset["top_left"],
        preset["bottom_left"],
        preset["top_right"],
        preset["bottom_right"],
    ])


# This function retieves the perspective preset for a given media file name. If name is not found or given, then default preset
# is returned.
def get_perspective_preset_for_media(media_name):
    return MEDIA_PERSPECTIVE_PRESETS.get(media_name, DEFAULT_PERSPECTIVE_PRESET)


# This function retrieves the HSV presets for a given media file name.
def get_hsv_preset_for_media(media_name):
    return MEDIA_HSV_PRESETS.get(
        media_name,
        {
            "lower": DEFAULT_LOWER_HSV,
            "upper": DEFAULT_UPPER_HSV,
        },
    )

# This is the main function that processes a single video frame.
# It applies perspective transformation to get a bird's eye view, 
# then uses color thresholding in HSV space to create a binary mask of lane markings.
def process_frame(
    frame_bgr, # This is the input frame in BGR format
    lower_hsv=None,
    upper_hsv=None,
    perspective_preset=None,
    perspective_points=None,
    media_name=None,
    window_half_width=50,
    window_height=40,
):
    frame = cv2.resize(frame_bgr, FRAME_SIZE)
    width, height = FRAME_SIZE

    # Check if perspective points are provided directly
    if perspective_points is not None:
        pts1 = np.float32(perspective_points)
    # Otherwise, use the preset based on media name or default
    else:
        if perspective_preset is None:
            perspective_preset = get_perspective_preset_for_media(media_name)
        pts1 = _preset_points(perspective_preset)

    # Check if HSV thresholds are provided, otherwise use presets based on media name or default
    if lower_hsv is None or upper_hsv is None:
        hsv_preset = get_hsv_preset_for_media(media_name)
        lower_hsv = hsv_preset["lower"]
        upper_hsv = hsv_preset["upper"]

    # Define the destination points for perspective transformation (bird's eye view)
    pts2 = np.float32([[0, 0], [0, height], [width, 0], [width, height]])

    # Compute the perspective transformation matrix and apply it to the input frame to get the bird's eye view
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_frame = cv2.warpPerspective(frame, matrix, FRAME_SIZE)

    # Here we convert the transformed frame to HSV color space and apply color thresholding to create a binary mask of 
    # lane markings.
    hsv_transformed_frame = cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv_transformed_frame,
        np.array(lower_hsv, dtype=np.uint8),
        np.array(upper_hsv, dtype=np.uint8),
    )

    # The following code implements a sliding window approach to detect lane markings in the binary mask. It starts from 
    # the bottom of the image and moves upwards, looking for contours in the left and right windows defined by the current 
    # base positions. The detected lane points are collected in lx, ly for the left lane and rx, ry for the right lane. The 
    # sliding windows are also drawn on a copy of the mask for visualization.
    histogram = np.sum(mask[mask.shape[0] // 2 :, :], axis=0)
    midpoint = int(histogram.shape[0] / 2)
    left_base = int(np.argmax(histogram[:midpoint]))
    right_base = int(np.argmax(histogram[midpoint:]) + midpoint)

    y = height - 8
    lx, ly = [], []
    rx, ry = [], []
    sliding_windows = cv2.cvtColor(mask.copy(), cv2.COLOR_GRAY2BGR)

    # We use a while loop to move the sliding windows upwards until we reach the top of the image.
    # We start from the bottom of the image and decrease y by window height in each iteration.
    while y > 0:
        # Here we define the top boundary of the sliding windows. We ensure that we don't go above the top of the image.
        top = max(0, y - window_height)

        #Here we define the left and right sliding windows based on the current base positions and the specified half width. 
        # We also ensure that we don't go outside the image boundaries.
        # Max and min ensure that the window stays within the image boundaries.
        left_start = max(0, left_base - window_half_width) # Left base is the current x position of the left lane.
        left_end = min(width, left_base + window_half_width) # Window half width defines how wide the search window is.
        # Here we crop the left window from the mask using the defined boundaries.
        # We take rows from top to y and columns from left start to left end.
        left_window = mask[top:y, left_start:left_end]
        # This function finds the contours in the left window. These are possible lane markings.
        contours, _ = cv2.findContours(left_window, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            moments = cv2.moments(contour)
            # mm0 is simply the area of the contour. It should not be zero to prevent divison by zero.
            if moments["m00"] != 0:
                cx = int(moments["m10"] / moments["m00"])
                cy = int(moments["m01"] / moments["m00"])
                left_base = left_start + cx
                lx.append(left_base)
                ly.append(top + cy)

        # Same process for the right wnidow
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

        # Here we draw a blue rectangle around the current left and right search windows.
        cv2.rectangle(
            sliding_windows,
            (max(0, left_base - window_half_width), y),
            (min(width, left_base + window_half_width), top),
            (255, 0, 0), # Blue color in BGR
            2, # Thickness
        )
        cv2.rectangle(
            sliding_windows,
            (max(0, right_base - window_half_width), y),
            (min(width, right_base + window_half_width), top),
            (255, 0, 0),
            2,
        )

        # Now we move to the next level by moving upward.
        y -= window_height

    # Here we create a copy of the original frame to draw the detected lanes.
    annotated = frame.copy()

    # Here we check if we have enough detected points for lanes.
    if len(lx) >= 4 and len(rx) >= 4:

        # Here we compute the inverse perspective transformation matrix. Earlier, the road was transformed into bird's-eye 
        # view. This inverse matrix maps the lane overlay back to the original camera view.
        inv_matrix = cv2.getPerspectiveTransform(pts2, pts1)

        # Here we create a blank image with the same size as the bird's-eye-view frame. Lane drawings will be placed on 
        # this overlay.
        overlay = np.zeros_like(transformed_frame)

        # Here we fit a first degree polynomial to the detected lane points.
        left_fit = np.polyfit(ly, lx, 2)
        right_fit = np.polyfit(ry, rx, 2)

        # Since the lanes are mostly vertical, it is better to predict x from y.
        # Here we get all y values from top to the bottom of the image.
        y_range = np.arange(0, height)
        # Here we calculate x values from the fitten polynomial, convert it to integer pixel coordinates and keep x values
        # within the image boundaries using np.clip.
        left_x = np.clip(np.polyval(left_fit, y_range).astype(int), 0, width - 1)
        right_x = np.clip(np.polyval(right_fit, y_range).astype(int), 0, width - 1)

        # Here we create coordinate pairs for the left and right lanes.
        pts_left = np.array([left_x, y_range]).T
        pts_right = np.array([right_x, y_range]).T

        # Here we create a filled polygon between left and reverse of right lane.
        fill_poly = np.vstack([pts_left, pts_right[::-1]]).astype(np.int32)
        cv2.fillPoly(overlay, [fill_poly], (0, 200, 0))

        # Reshape is required for OpenCV drawing functions.
        left_draw = pts_left.reshape(-1, 1, 2).astype(np.int32)
        right_draw = pts_right.reshape(-1, 1, 2).astype(np.int32)

        # Draw lane lines.
        cv2.polylines(overlay, [left_draw], False, (0, 0, 255), 5)
        cv2.polylines(overlay, [right_draw], False, (255, 0, 0), 5)

        # We warp the lane overlay from bird's-eye view back to the original camera perspective.
        inv_overlay = cv2.warpPerspective(overlay, inv_matrix, FRAME_SIZE)
        # Blend the overlay with the original frame.
        annotated = cv2.addWeighted(annotated, 1.0, inv_overlay, 0.5, 0)

    # Draw red circles on the original frame showing the perspective transform source points.
    for point in pts1.astype(int):
        cv2.circle(annotated, tuple(point), 5, (0, 0, 255), -1)

    # Return the results.
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
