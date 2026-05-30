import sys
import importlib
import json
import re
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
import pandas as pd
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
import ImageProcessingTechnique.processor as image_processor

image_processor = importlib.reload(image_processor)
FRAME_SIZE = image_processor.FRAME_SIZE
PERSPECTIVE_PRESETS = image_processor.PERSPECTIVE_PRESETS
get_hsv_preset_for_media = image_processor.get_hsv_preset_for_media
get_perspective_preset_for_media = image_processor.get_perspective_preset_for_media
process_frame = image_processor.process_frame

st.set_page_config(page_title="Lane Detection", layout="wide")
st.title("Real Time Lane Detection")

CHECKPOINT_DIR = APP_DIR / "checkpoints"
RECOMMENDED_CHECKPOINT = "3.0_Pillar.pt"
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


def format_number(value):
    return f"{value:,}"


def count_parameters(model):
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


def checkpoint_history_path(checkpoint_path):
    version = checkpoint_path.stem.split("_", 1)[0]
    return checkpoint_path.with_name(f"{version}_History.json")


@st.cache_data
def count_parameters_for_architecture(architecture):
    if architecture == "LaneNet":
        model = LaneNet(embedding_dim=4)
    else:
        model = LaneNetResNet34(embedding_dim=4, pretrained=False)
    return count_parameters(model)


@st.cache_data
def load_checkpoint_metadata(checkpoint_path):
    path = Path(checkpoint_path)
    ckpt = torch.load(path, map_location="cpu")
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    state_dict = ckpt.get("model_state_dict", ckpt)
    first_key = next(iter(state_dict), "")
    architecture = "LaneNet" if first_key.startswith("encoder.initial_block") else "LaneNetResNet34"
    total_params, trainable_params = count_parameters_for_architecture(architecture)
    return {
        "checkpoint": path.name,
        "path": str(path),
        "architecture": architecture,
        "epoch": epoch,
        "size_mb": path.stat().st_size / (1024 * 1024),
        "parameters": total_params,
        "trainable_parameters": trainable_params,
    }


@st.cache_data
def load_checkpoint_history(history_path):
    path = Path(history_path)
    if not path.exists():
        return None

    with path.open("r", encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        return None

    clean_rows = [
        row
        for row in rows
        if isinstance(row, dict) and row.get("epoch") is not None
    ]
    if not clean_rows:
        return None

    best = max(clean_rows, key=lambda row: row.get("val_iou", float("-inf")))
    lowest_val_loss = min(
        clean_rows,
        key=lambda row: row.get("val_total", float("inf")),
    )
    return {
        "path": path.name,
        "rows": clean_rows,
        "best": best,
        "latest": clean_rows[-1],
        "lowest_val_loss": lowest_val_loss,
    }


def build_model_comparison(checkpoint_paths):
    rows = []
    curves = {}

    for checkpoint_path in checkpoint_paths:
        metadata = load_checkpoint_metadata(str(checkpoint_path))
        history_path = checkpoint_history_path(checkpoint_path)
        history = load_checkpoint_history(str(history_path))
        is_recommended = metadata["checkpoint"] == RECOMMENDED_CHECKPOINT
        row = {
            "Checkpoint": metadata["checkpoint"],
            "Recommended": is_recommended,
            "Architecture": metadata["architecture"],
            "Parameters": metadata["parameters"],
            "Checkpoint epoch": metadata["epoch"],
            "Size MB": metadata["size_mb"],
            "History": history["path"] if history else "Missing",
            "History epochs": len(history["rows"]) if history else None,
            "Best val IoU": history["best"].get("val_iou") if history else None,
            "Best IoU epoch": history["best"].get("epoch") if history else None,
            "Lowest val loss": history["lowest_val_loss"].get("val_total") if history else None,
            "Latest val IoU": history["latest"].get("val_iou") if history else None,
            "Latest val loss": history["latest"].get("val_total") if history else None,
        }
        rows.append(row)

        if history:
            history_frame = pd.DataFrame(history["rows"]).set_index("epoch")
            if "val_iou" in history_frame:
                curves[f"{checkpoint_path.stem} val_iou"] = history_frame["val_iou"]
            if "val_total" in history_frame:
                curves[f"{checkpoint_path.stem} val_total"] = history_frame["val_total"]

    return pd.DataFrame(rows), curves


def show_model_comparison(checkpoint_paths, selected_checkpoint):
    comparison_frame, curves = build_model_comparison(checkpoint_paths)
    if comparison_frame.empty:
        return

    with st.expander("Model comparison", expanded=False):
        best_iou = comparison_frame["Best val IoU"].max(skipna=True)
        best_loss = comparison_frame["Lowest val loss"].min(skipna=True)
        best_iou_row = comparison_frame.loc[
            comparison_frame["Best val IoU"].idxmax()
        ] if not pd.isna(best_iou) else None
        best_loss_row = comparison_frame.loc[
            comparison_frame["Lowest val loss"].idxmin()
        ] if not pd.isna(best_loss) else None

        recommended_row = comparison_frame[
            comparison_frame["Checkpoint"].eq(RECOMMENDED_CHECKPOINT)
        ]
        recommended_label = (
            RECOMMENDED_CHECKPOINT
            if not recommended_row.empty
            else "N/A"
        )

        compare_cols = st.columns(4)
        compare_cols[0].metric("Checkpoints", len(comparison_frame))
        compare_cols[1].metric(
            "Best working model",
            recommended_label,
        )
        compare_cols[2].metric(
            "Best model by IoU",
            best_iou_row["Checkpoint"] if best_iou_row is not None else "N/A",
            f"{best_iou:.4f}" if best_iou_row is not None else None,
        )
        compare_cols[3].metric(
            "Lowest validation loss",
            best_loss_row["Checkpoint"] if best_loss_row is not None else "N/A",
            f"{best_loss:.3f}" if best_loss_row is not None else None,
        )

        st.caption(
            "`3.0_Pillar.pt` is marked as the best working model based on practical "
            "lane-detection behavior, even when another checkpoint has a higher validation metric."
        )

        display_frame = comparison_frame.copy()
        display_frame.insert(
            0,
            "Selected",
            display_frame["Checkpoint"].eq(selected_checkpoint.name),
        )
        st.dataframe(
            display_frame,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Selected": st.column_config.CheckboxColumn("Selected"),
                "Recommended": st.column_config.CheckboxColumn("Recommended"),
                "Parameters": st.column_config.NumberColumn("Parameters", format="%d"),
                "Size MB": st.column_config.NumberColumn("Size MB", format="%.1f"),
                "Best val IoU": st.column_config.NumberColumn("Best val IoU", format="%.4f"),
                "Lowest val loss": st.column_config.NumberColumn("Lowest val loss", format="%.3f"),
                "Latest val IoU": st.column_config.NumberColumn("Latest val IoU", format="%.4f"),
                "Latest val loss": st.column_config.NumberColumn("Latest val loss", format="%.3f"),
            },
        )

        if curves:
            curve_frame = pd.DataFrame(curves)
            iou_columns = [column for column in curve_frame.columns if column.endswith("val_iou")]
            loss_columns = [column for column in curve_frame.columns if column.endswith("val_total")]
            if iou_columns:
                st.line_chart(curve_frame[iou_columns])
            if loss_columns:
                st.line_chart(curve_frame[loss_columns])


@st.cache_data
def load_training_summary():
    notebook_path = DEEP_LEARNING_DIR / "trainCULane.ipynb"
    if not notebook_path.exists():
        return {
            "notebook": None,
            "train_samples": None,
            "val_samples": None,
            "train_batches": None,
            "val_batches": None,
            "best_epoch": None,
            "best_iou": None,
            "latest": None,
        }

    text = notebook_path.read_text(encoding="utf-8", errors="ignore")
    dataset_values = {}
    for key in ["train samples", "val samples", "train batches", "val batches"]:
        match = re.search(rf"{re.escape(key)}:\s*(\d+)", text)
        dataset_values[key.replace(" ", "_")] = int(match.group(1)) if match else None

    metric_pattern = re.compile(
        r"\[(\d+)/(\d+)\]\s+"
        r"train_total=([0-9.]+)\s+"
        r"train_bin=([0-9.]+)\s+"
        r"train_disc=([0-9.]+)\s+\|\s+"
        r"val_total=([0-9.]+)\s+"
        r"val_bin=([0-9.]+)\s+"
        r"val_disc=([0-9.]+)\s+"
        r"val_iou=([0-9.]+)"
    )
    rows = [
        {
            "epoch": int(match.group(1)),
            "total_epochs": int(match.group(2)),
            "train_total": float(match.group(3)),
            "train_binary": float(match.group(4)),
            "train_discriminative": float(match.group(5)),
            "val_total": float(match.group(6)),
            "val_binary": float(match.group(7)),
            "val_discriminative": float(match.group(8)),
            "val_iou": float(match.group(9)),
        }
        for match in metric_pattern.finditer(text)
    ]

    best = max(rows, key=lambda row: row["val_iou"], default=None)
    return {
        "notebook": notebook_path.name,
        **dataset_values,
        "best_epoch": best["epoch"] if best else None,
        "best_iou": best["val_iou"] if best else None,
        "latest": rows[-1] if rows else None,
    }


def show_deep_learning_model_details(
    model,
    architecture,
    device,
    checkpoint_path,
    checkpoint_epoch,
):
    total_params, trainable_params = count_parameters(model)
    checkpoint_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    history_path = checkpoint_history_path(checkpoint_path)
    checkpoint_history = load_checkpoint_history(str(history_path))
    training_summary = load_training_summary()

    with st.expander("Deep learning model details", expanded=False):
        st.subheader("Model features")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Architecture", architecture)
        col2.metric("Parameters", format_number(total_params))
        col3.metric("Trainable", format_number(trainable_params))
        col4.metric("Checkpoint epoch", checkpoint_epoch or "N/A")

        st.caption(
            f"Checkpoint: `{checkpoint_path.name}` | "
            f"Size: `{checkpoint_size_mb:.1f} MB` | Device: `{device}`"
        )

        st.markdown("**Architecture summary**")
        if architecture == "LaneNetResNet34":
            st.write(
                "ResNet34 encoder with U-Net style decoder heads. One decoder predicts "
                "binary lane/background logits, and the second decoder predicts a "
                "4-channel instance embedding map for separating lane instances."
            )
        else:
            st.write(
                "ENet-style LaneNet encoder with two independent decoder heads: binary "
                "segmentation and 4-channel instance embedding."
            )

        if checkpoint_history:
            st.markdown("**Training results from checkpoint history**")
            best = checkpoint_history["best"]
            latest = checkpoint_history["latest"]
            lowest_val_loss = checkpoint_history["lowest_val_loss"]

            metric_cols = st.columns(4)
            metric_cols[0].metric("History epochs", len(checkpoint_history["rows"]))
            metric_cols[1].metric("Best validation IoU", f"{best['val_iou']:.4f}")
            metric_cols[2].metric("Best IoU epoch", best["epoch"])
            metric_cols[3].metric(
                "Lowest validation loss",
                f"{lowest_val_loss['val_total']:.3f}",
            )

            st.table(
                [
                    {"Metric": "Latest epoch", "Value": latest["epoch"]},
                    {"Metric": "Latest train total loss", "Value": f"{latest['train_total']:.3f}"},
                    {"Metric": "Latest validation total loss", "Value": f"{latest['val_total']:.3f}"},
                    {"Metric": "Latest validation IoU", "Value": f"{latest['val_iou']:.4f}"},
                    {"Metric": "Best train total loss", "Value": f"{best['train_total']:.3f}"},
                    {"Metric": "Best validation total loss", "Value": f"{best['val_total']:.3f}"},
                    {"Metric": "Best validation binary loss", "Value": f"{best['val_binary']:.3f}"},
                    {"Metric": "Best validation discriminative loss", "Value": f"{best['val_disc']:.3f}"},
                ]
            )

            chart_rows = pd.DataFrame(checkpoint_history["rows"]).set_index("epoch")
            loss_columns = [
                column
                for column in ["train_total", "val_total", "train_binary", "val_binary"]
                if column in chart_rows
            ]
            if loss_columns:
                st.line_chart(chart_rows[loss_columns])
            if "val_iou" in chart_rows:
                st.line_chart(chart_rows[["val_iou"]])

            st.caption(f"History source: `{checkpoint_history['path']}`")
            return

        st.markdown("**Training results found in notebook logs**")
        metric_cols = st.columns(4)
        metric_cols[0].metric(
            "Train samples",
            format_number(training_summary["train_samples"])
            if training_summary["train_samples"]
            else "N/A",
        )
        metric_cols[1].metric(
            "Validation samples",
            format_number(training_summary["val_samples"])
            if training_summary["val_samples"]
            else "N/A",
        )
        metric_cols[2].metric(
            "Best validation IoU",
            f"{training_summary['best_iou']:.4f}"
            if training_summary["best_iou"] is not None
            else "N/A",
        )
        metric_cols[3].metric(
            "Best epoch",
            training_summary["best_epoch"] or "N/A",
        )

        latest = training_summary["latest"]
        if latest:
            st.table(
                [
                    {"Metric": "Latest logged epoch", "Value": latest["epoch"]},
                    {"Metric": "Train total loss", "Value": f"{latest['train_total']:.3f}"},
                    {"Metric": "Validation total loss", "Value": f"{latest['val_total']:.3f}"},
                    {"Metric": "Validation IoU", "Value": f"{latest['val_iou']:.4f}"},
                ]
            )
        else:
            st.info("No epoch-by-epoch training metrics were found in the notebook logs.")

        st.caption(
            "The selected checkpoint stores model weights, optimizer state, and epoch. "
            "Detailed validation metrics are read from the available training notebook logs."
        )


def read_uploaded_image(uploaded_file):
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    return cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)


def get_image_processing_settings(media_name, key_prefix):
    width, height = FRAME_SIZE
    preset_name = get_perspective_preset_for_media(media_name)
    perspective_defaults = PERSPECTIVE_PRESETS[preset_name]
    hsv_defaults = get_hsv_preset_for_media(media_name)

    st.sidebar.subheader("Image processing controls")
    st.sidebar.caption(f"Defaults: `{preset_name}` perspective preset")

    lower_default = hsv_defaults["lower"]
    upper_default = hsv_defaults["upper"]
    with st.sidebar.expander("HSV thresholds", expanded=True):
        st.caption("Lower HSV")
        lower_h = st.slider(
            "Lower H",
            0,
            179,
            int(lower_default[0]),
            key=f"{key_prefix}_lower_h",
        )
        lower_s = st.slider(
            "Lower S",
            0,
            255,
            int(lower_default[1]),
            key=f"{key_prefix}_lower_s",
        )
        lower_v = st.slider(
            "Lower V",
            0,
            255,
            int(lower_default[2]),
            key=f"{key_prefix}_lower_v",
        )

        st.caption("Upper HSV")
        upper_h = st.slider(
            "Upper H",
            0,
            179,
            int(upper_default[0]),
            key=f"{key_prefix}_upper_h",
        )
        upper_s = st.slider(
            "Upper S",
            0,
            255,
            int(upper_default[1]),
            key=f"{key_prefix}_upper_s",
        )
        upper_v = st.slider(
            "Upper V",
            0,
            255,
            int(upper_default[2]),
            key=f"{key_prefix}_upper_v",
        )

    perspective_points = []
    point_labels = {
        "top_left": "Top left",
        "bottom_left": "Bottom left",
        "top_right": "Top right",
        "bottom_right": "Bottom right",
    }
    with st.sidebar.expander("Perspective source points", expanded=True):
        for point_key, label in point_labels.items():
            default_x, default_y = perspective_defaults[point_key]
            st.caption(label)
            x_value = st.slider(
                f"{label} X",
                0,
                width - 1,
                int(default_x),
                key=f"{key_prefix}_{point_key}_x",
            )
            y_value = st.slider(
                f"{label} Y",
                0,
                height - 1,
                int(default_y),
                key=f"{key_prefix}_{point_key}_y",
            )
            perspective_points.append((x_value, y_value))

    return {
        "lower_hsv": (lower_h, lower_s, lower_v),
        "upper_hsv": (upper_h, upper_s, upper_v),
        "perspective_points": perspective_points,
    }


model = None
device = "cpu"
architecture = None
selected_checkpoint = None
clustering_mode = "embedding_spatial"
horizon_padding = 20

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
    use_spatial_clustering = st.sidebar.toggle(
        "Use spatial coordinates for clustering",
        value=True,
        help=(
            "When enabled, DBSCAN uses embedding vectors together with normalized "
            "x/y lane-pixel coordinates. When disabled, DBSCAN uses only the "
            "network embedding vectors."
        ),
    )
    clustering_mode = (
        "embedding_spatial" if use_spatial_clustering else "embedding_only"
    )
    horizon_padding = st.sidebar.slider(
        "Horizon padding",
        min_value=0,
        max_value=120,
        value=20,
        step=5,
        help=(
            "Extra rows ignored below the detected horizon before clustering. "
            "Increasing this can remove noisy far-away lane pixels."
        ),
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
    checkpoint_epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
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
    return model, device, architecture, checkpoint_epoch

if technique == "Deep learning":
    model, device, architecture, checkpoint_epoch = load_model(str(selected_checkpoint))
    st.sidebar.info(f"Loaded `{selected_checkpoint.name}` ({architecture}) on `{device}`")
    show_deep_learning_model_details(
        model,
        architecture,
        device,
        selected_checkpoint,
        checkpoint_epoch,
    )
    show_model_comparison(get_available_checkpoints(), selected_checkpoint)

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


def run_image_processing(
    frame_bgr,
    media_name=None,
    lower_hsv=None,
    upper_hsv=None,
    perspective_points=None,
):
    return process_frame(
        frame_bgr,
        media_name=media_name,
        lower_hsv=lower_hsv,
        upper_hsv=upper_hsv,
        perspective_points=perspective_points,
    )


@torch.no_grad()
def run_inference(
    frame_bgr,
    previous_ego_selection=None,
    lane_departure_enabled=True,
    clustering_mode="embedding_spatial",
    horizon_padding=20,
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
        horizon_padding=horizon_padding,
        eps=0.35,
        min_samples=20,
        spatial_weight=0.5,
        clustering_mode=clustering_mode,
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

    image_processing_settings = None
    if technique == "Image processing" and frame_bgr is not None:
        settings_media_key = Path(media_name or "uploaded_image").stem
        image_processing_settings = get_image_processing_settings(
            media_name,
            f"image_processing_image_{settings_media_key}",
        )

    if frame_bgr is not None:
        if technique == "Deep learning":
            annotated, lanes, binary_rgb, model_outputs = run_inference(
                frame_bgr,
                lane_departure_enabled=enable_lane_departure,
                clustering_mode=clustering_mode,
                horizon_padding=horizon_padding,
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
            outputs = run_image_processing(
                frame_bgr,
                media_name=media_name,
                **image_processing_settings,
            )
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

    image_processing_settings = None
    if technique == "Image processing" and video_path:
        settings_media_key = Path(media_name or "uploaded_video").stem
        image_processing_settings = get_image_processing_settings(
            media_name,
            f"image_processing_video_{settings_media_key}",
        )

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
                        clustering_mode=clustering_mode,
                        horizon_padding=horizon_padding,
                        media_name=media_name,
                    )
                    previous_ego_selection = model_outputs["ego_selection"]
                else:
                    outputs = run_image_processing(
                        frame,
                        media_name=media_name,
                        **image_processing_settings,
                    )
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
