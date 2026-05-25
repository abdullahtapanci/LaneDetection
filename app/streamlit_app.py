import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
APP_DIR = Path(__file__).resolve().parent
DEEP_LEARNING_DIR = ROOT_DIR / "DeepLearningTechnique"
sys.path.insert(0, str(APP_DIR))
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(DEEP_LEARNING_DIR))

import streamlit as st
import cv2
import numpy as np
import torch
import time
from lane_warner import (
    LaneDepartureState,
    assess_lane_departure,
    draw_lane_departure_warning,
    select_ego_lanes,
)
from DeepLearningTechnique.src.models.lanenet import LaneNet, LaneNetResNet34
from DeepLearningTechnique.src.postprocess import my_postprocess, draw_lanes
from ImageProcessingTechnique.processor import process_frame

st.set_page_config(page_title="Lane Detection", layout="wide")
st.title("Real Time Lane Detection")

CHECKPOINT_DIR = APP_DIR / "checkpoints"
EXAMPLE_IMAGE_DIR = ROOT_DIR / "exampleImages"
EXAMPLE_VIDEO_DIR = ROOT_DIR / "exampleVideos"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
technique = st.sidebar.radio(
    "Technique",
    ["Deep learning", "Image processing"],
)


def get_available_checkpoints():
    return sorted(CHECKPOINT_DIR.glob("*.pt"), key=lambda path: path.stem)


def get_example_files(directory, extensions):
    if not directory.exists():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def read_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


model = None
device = "cpu"
architecture = None
selected_checkpoint = None

if technique == "Deep learning":
    # ───────────────────────── 1. Select and load model ─────────────────────────
    checkpoint_paths = get_available_checkpoints()
    if not checkpoint_paths:
        st.error(f"No checkpoint files found in `{CHECKPOINT_DIR}`.")
        st.stop()

    selected_checkpoint = st.sidebar.selectbox(
        "Model checkpoint",
        checkpoint_paths,
        index=0,
        format_func=lambda path: path.stem.replace("_", " ").title(),
    )
    enable_lane_departure = st.sidebar.toggle("Lane departure", value=True)
else:
    enable_lane_departure = False

@st.cache_resource
def load_model(ckpt_path):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    first_key = next(iter(state_dict), "")

    if first_key.startswith("encoder.initial_block"):
        model = LaneNet(embedding_dim=4)
        architecture = "LaneNet"
    else:
        model = LaneNetResNet34(embedding_dim=4, pretrained=False)
        architecture = "LaneNetResNet34"

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device, architecture

if technique == "Deep learning":
    model, device, architecture = load_model(str(selected_checkpoint))
    st.sidebar.info(f"Loaded `{selected_checkpoint.name}` ({architecture}) on `{device}`")

# ───────────────────────── 3. Preprocess + inference helpers ─────────────────────────
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_W, TARGET_H = 768, 384

MIN_BLOB_PIXELS = 10

def draw_ego_region(image_rgb, lanes, left_idx, right_idx, alpha=0.25):
    """Fill the area between the two ego lanes with a semi-transparent green overlay."""
    if left_idx is None or right_idx is None:
        return image_rgb
    H, W  = image_rgb.shape[:2]
    ll, rl = lanes[left_idx], lanes[right_idx]
    y_min = max(ll['y_min'], rl['y_min'])
    y_max = min(ll['y_max'], rl['y_max'])
    if y_min >= y_max:
        return image_rgb
    ys       = np.arange(y_min, y_max + 1, 2)
    xs_left  = np.clip(np.polyval(ll['poly'], ys).astype(int), 0, W - 1)
    xs_right = np.clip(np.polyval(rl['poly'], ys).astype(int), 0, W - 1)
    pts      = np.concatenate([
        np.stack([xs_left,  ys], axis=1),
        np.stack([xs_right, ys], axis=1)[::-1],
    ]).reshape(-1, 1, 2)
    overlay = image_rgb.copy()
    cv2.fillPoly(overlay, [pts], (180, 180, 180))
    return cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)


def draw_single_lane(image_rgb, lane, colour, thickness=3, point_step=4):
    """Draw one lane polynomial onto image_rgb in the given colour."""
    H, W = image_rgb.shape[:2]
    ys = np.arange(max(0, lane['y_min']), min(H, lane['y_max'] + 1), point_step)
    if len(ys) < 2:
        return image_rgb
    xs = np.polyval(lane['poly'], ys).astype(int)
    for j in range(len(ys) - 1):
        x0, y0 = int(np.clip(xs[j],     0, W - 1)), int(ys[j])
        x1, y1 = int(np.clip(xs[j + 1], 0, W - 1)), int(ys[j + 1])
        cv2.line(image_rgb, (x0, y0), (x1, y1), colour, thickness, cv2.LINE_AA)
    return image_rgb


def build_lane_canvas(frame_rgb, lanes, left_idx, right_idx, departure_state):
    lane_canvas = np.full_like(frame_rgb, 255)
    lane_canvas = draw_ego_region(lane_canvas, lanes, left_idx, right_idx, alpha=0.6)
    lane_canvas = draw_lane_departure_warning(
        lane_canvas,
        lanes,
        left_idx,
        right_idx,
        departure_state,
    )
    for lane in lanes:
        draw_single_lane(lane_canvas, lane, colour=(0, 0, 0), thickness=3)
    return lane_canvas


def preprocess(frame_bgr):
    frame = cv2.resize(frame_bgr, (TARGET_W, TARGET_H))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = (frame_rgb.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(f).permute(2,0,1).unsqueeze(0).to(device), frame_rgb


def binary_mask_image(binary_logits):
    mask = binary_logits.argmax(axis=0).astype(np.uint8) * 255
    return np.stack([mask, mask, mask], axis=-1)


def lane_probability_map(binary_logits):
    logits = binary_logits - binary_logits.max(axis=0, keepdims=True)
    probs = np.exp(logits) / np.exp(logits).sum(axis=0, keepdims=True)
    lane_prob = probs[1]
    heatmap = (np.clip(lane_prob, 0.0, 1.0) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), lane_prob


def embedding_image(embedding):
    channels = embedding[:3]
    if channels.shape[0] < 3:
        channels = np.pad(channels, ((0, 3 - channels.shape[0]), (0, 0), (0, 0)))

    channels = np.moveaxis(channels, 0, -1)
    low, high = np.percentile(channels, [2, 98])
    if np.isclose(low, high):
        return np.zeros((*embedding.shape[1:], 3), dtype=np.uint8)

    channels = (channels - low) / (high - low)
    return (np.clip(channels, 0.0, 1.0) * 255).astype(np.uint8)


def run_image_processing(frame_bgr, media_name=None):
    return process_frame(frame_bgr, media_name=media_name)


@torch.no_grad()
def run_inference(
    frame_bgr,
    previous_ego_selection=None,
    lane_departure_enabled=True,
    media_name=None,
):
    x, frame_rgb = preprocess(frame_bgr)
    binary_logits, embedding = model(x)
    binary_np = binary_logits.squeeze(0).cpu().numpy()
    emb_np    = embedding.squeeze(0).cpu().numpy()
    binary_output = binary_mask_image(binary_np)
    probability_output, lane_prob = lane_probability_map(binary_np)
    embedding_output = embedding_image(emb_np)

    lanes = my_postprocess(
        binary_np,
        emb_np,
        min_blob_pixels=MIN_BLOB_PIXELS,
        lane_probability_threshold=0.95,
        horizon_min_pixels=12,
        horizon_padding=20,
        eps=0.35,
        min_samples=20,
        spatial_weight=0.5,
        min_pixels=100,
        poly_degree=2
    )
    H, W = frame_rgb.shape[:2]
    annotated = draw_lanes(frame_rgb.copy(), lanes)
    ego_selection = select_ego_lanes(lanes, W, H)
    left_idx, right_idx = ego_selection.left_idx, ego_selection.right_idx

    if lane_departure_enabled:
        departure_state = assess_lane_departure(lanes, left_idx, right_idx, W, H)
    else:
        departure_state = LaneDepartureState(
            False,
            None,
            None,
            "Lane departure disabled",
        )

    lane_canvas = build_lane_canvas(
        frame_rgb,
        lanes,
        left_idx,
        right_idx,
        departure_state,
    )

    model_outputs = {
        "binary": binary_output,
        "embedding": embedding_output,
        "lane_probability": probability_output,
        "lane_probability_values": lane_prob,
        "departure": departure_state,
        "ego_selection": ego_selection,
    }
    return annotated, lanes, lane_canvas, model_outputs

# ───────────────────────── 4. UI: choose image or video ─────────────────────────
mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

if mode == "Image":
    image_source = st.radio(
        "Image source",
        ["Example image", "Upload image"],
        horizontal=True,
    )
    frame_bgr = None
    media_name = None

    if image_source == "Example image":
        example_images = get_example_files(EXAMPLE_IMAGE_DIR, IMAGE_EXTENSIONS)
        if example_images:
            selected_image = st.selectbox(
                "Example image",
                example_images,
                format_func=lambda path: path.name,
            )
            frame_bgr = cv2.imread(str(selected_image))
            media_name = selected_image.name
        else:
            st.info(f"No example images found in `{EXAMPLE_IMAGE_DIR}`.")
    else:
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp", "bmp"])
        if uploaded:
            frame_bgr = read_uploaded_image(uploaded)

    if frame_bgr is not None:

        if technique == "Deep learning":
            annotated, lanes, binary_rgb, model_outputs = run_inference(
                frame_bgr,
                lane_departure_enabled=enable_lane_departure,
                media_name=media_name,
            )
            col1, col2, col3 = st.columns(3)
            col1.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Input")
            col2.image(binary_rgb, caption="Lane canvas")
            col3.image(annotated, caption=f"Detected {len(lanes)} lanes")

            out1, out2, out3 = st.columns(3)
            out1.image(model_outputs["binary"], caption="Binary output")
            out2.image(model_outputs["embedding"], caption="Embedding output")
            out3.image(
                model_outputs["lane_probability"],
                caption="Lane pixel probability distribution",
            )
            st.caption(
                "Lane probability range: "
                f"{model_outputs['lane_probability_values'].min():.3f} - "
                f"{model_outputs['lane_probability_values'].max():.3f}"
            )
        else:
            outputs = run_image_processing(frame_bgr, media_name=media_name)
            col1, col2 = st.columns(2)
            col1.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Input")
            col2.image(outputs["annotated"], caption="Image processing result")

            out1, out2, out3 = st.columns(3)
            out1.image(outputs["bird_eye"], caption="Bird's-eye view")
            out2.image(outputs["mask"], caption="HSV threshold mask")
            out3.image(outputs["sliding_windows"], caption="Sliding windows")
            points = outputs["detected_points"]
            st.caption(
                f"Detected window points: left={points['left']} right={points['right']}"
            )

else:   # Video
    video_source = st.radio(
        "Video source",
        ["Example video", "Upload video"],
        horizontal=True,
    )
    video_path = None
    media_name = None

    if video_source == "Example video":
        example_videos = get_example_files(EXAMPLE_VIDEO_DIR, VIDEO_EXTENSIONS)
        if example_videos:
            selected_video = st.selectbox(
                "Example video",
                example_videos,
                format_func=lambda path: path.name,
            )
            video_path = str(selected_video)
            media_name = selected_video.name
            st.video(video_path)
        else:
            st.info(f"No example videos found in `{EXAMPLE_VIDEO_DIR}`.")
    else:
        uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
        if uploaded:
            tmp_in = "/tmp/streamlit_in.mp4"
            with open(tmp_in, "wb") as f:
                f.write(uploaded.read())
            video_path = tmp_in
            media_name = uploaded.name

    if video_path:
        if st.button("Process video"):
            cap = cv2.VideoCapture(video_path)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0

            tmp_out = "/tmp/streamlit_out.mp4"
            writer  = None

            left_col, right_col = st.columns([0.8, 2], vertical_alignment="center")
            raw_preview       = left_col.empty()
            binary_preview    = left_col.empty()
            probability_preview = left_col.empty()
            embedding_preview = left_col.empty()
            annotated_preview = right_col.empty()
            canvas_preview    = right_col.empty()

            stats_bar = st.empty()
            progress  = st.progress(0)

            i              = 0
            fps_window     = []           # rolling window of per-frame times
            WINDOW_SIZE    = 10          # smooth FPS over last 10 frames
            previous_ego_selection = None
            current_fps = 0.0

            while True:
                t0 = time.perf_counter()

                ok, frame = cap.read()
                if not ok:
                    break

                raw_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if technique == "Deep learning":
                    annotated, _, lane_canvas, model_outputs = run_inference(
                        frame,
                        previous_ego_selection,
                        lane_departure_enabled=enable_lane_departure,
                        media_name=media_name,
                    )
                    previous_ego_selection = model_outputs["ego_selection"]
                else:
                    outputs = run_image_processing(frame, media_name=media_name)
                    annotated = outputs["annotated"]
                    lane_canvas = outputs["bird_eye"]
                    model_outputs = {
                        "binary": outputs["mask"],
                        "embedding": outputs["sliding_windows"],
                        "lane_probability": outputs["annotated"],
                    }
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)

                if writer is None:
                    h, w = annotated_bgr.shape[:2]
                    writer = cv2.VideoWriter(
                        tmp_out, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h)
                    )
                writer.write(annotated_bgr)

                # ── timing ────────────────────────────────────────────────
                elapsed = time.perf_counter() - t0
                fps_window.append(elapsed)
                if len(fps_window) > WINDOW_SIZE:
                    fps_window.pop(0)

                current_fps  = 1.0 / (sum(fps_window) / len(fps_window))
                frames_left  = total - (i + 1)
                eta_seconds  = frames_left / current_fps if current_fps > 0 else 0
                eta_str      = time.strftime("%M:%S", time.gmtime(eta_seconds))

                # ── update UI every 5 frames ──────────────────────────────
                if i % 5 == 0:
                    raw_preview.image(
                        raw_rgb,
                        caption=f"Raw  |  Frame {i+1}/{total}",
                        use_container_width=True
                    )
                    annotated_preview.image(
                        annotated,
                        caption=f"Annotated  |  Frame {i+1}/{total}  |  {current_fps:.1f} FPS",
                        use_container_width=True
                    )
                    canvas_preview.image(lane_canvas, caption="Lane canvas", use_container_width=True)
                    binary_preview.image(
                        model_outputs["binary"],
                        caption="Binary output" if technique == "Deep learning" else "HSV threshold mask",
                        use_container_width=True
                    )
                    embedding_preview.image(
                        model_outputs["embedding"],
                        caption="Embedding output" if technique == "Deep learning" else "Sliding windows",
                        use_container_width=True
                    )
                    probability_preview.image(
                        model_outputs["lane_probability"],
                        caption="Lane probability" if technique == "Deep learning" else "Image processing result",
                        use_container_width=True
                    )
                    stats_bar.markdown(
                        f"⚡ **Processing speed:** `{current_fps:.1f} FPS` &nbsp;&nbsp;"
                        f"🕐 **ETA:** `{eta_str}` &nbsp;&nbsp;"
                        f"📹 **Frame:** `{i+1} / {total}`"
                    )

                progress.progress((i + 1) / total)
                i += 1

            cap.release()
            if writer:
                writer.release()

            st.success(
                f"✅ Done — {i} frames processed at **{current_fps:.1f} FPS** average"
            )
            st.video(tmp_out)

            with open(tmp_out, "rb") as f:
                st.download_button("Download annotated video", f, "annotated.mp4", "video/mp4")
