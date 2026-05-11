# app/main.py
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))   # so we can import src.*

import streamlit as st
import cv2
import numpy as np
import torch
from src.models.lanenet import LaneNetResNet34
from src.postprocess import my_postprocess, draw_lanes

st.set_page_config(page_title="Lane Detection", layout="wide")
st.title("Lane Detection Demo")
st.write("If you can see this, Streamlit is running correctly.")

# ───────────────────────── 1. Load model once ─────────────────────────
@st.cache_resource
def load_model(ckpt_path="app/checkpoints/best.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LaneNetResNet34(embedding_dim=4, pretrained=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, device

model, device = load_model()
st.sidebar.info(f"Model loaded on `{device}`")

# ───────────────────────── 2. Sidebar controls ─────────────────────────
st.sidebar.header("Postprocess settings")
eps             = st.sidebar.slider("DBSCAN eps",       0.05, 1.0, 0.2, 0.05)
min_samples     = st.sidebar.slider("Min samples",      5, 100, 20, 5)
min_pixels      = st.sidebar.slider("Min cluster size", 20, 500, 100, 10)
min_blob_pixels = st.sidebar.slider("Min blob size",    5, 200, 30, 5)
poly_degree     = st.sidebar.selectbox("Polynomial degree", [1, 2, 3], index=1)

params = {
    'eps': eps, 'min_samples': min_samples, 'min_pixels': min_pixels,
    'min_blob_pixels': min_blob_pixels, 'poly_degree': poly_degree,
}

# ───────────────────────── 3. Preprocess + inference helpers ─────────────────────────
IMNET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMNET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
TARGET_W, TARGET_H = 768, 384

def preprocess(frame_bgr):
    frame = cv2.resize(frame_bgr, (TARGET_W, TARGET_H))
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    f = (frame_rgb.astype(np.float32) / 255.0 - IMNET_MEAN) / IMNET_STD
    return torch.from_numpy(f).permute(2,0,1).unsqueeze(0).to(device), frame_rgb

@torch.no_grad()
def run_inference(frame_bgr):
    x, frame_rgb = preprocess(frame_bgr)
    binary_logits, embedding = model(x)
    binary_np = binary_logits.squeeze(0).cpu().numpy()
    emb_np    = embedding.squeeze(0).cpu().numpy()
    lanes = my_postprocess(binary_np, emb_np, **params)
    annotated = draw_lanes(frame_rgb.copy(), lanes)
    return annotated, lanes

# ───────────────────────── 4. UI: choose image or video ─────────────────────────
mode = st.radio("Input type", ["Image", "Video"], horizontal=True)

if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        col1, col2 = st.columns(2)
        col1.image(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), caption="Input")
        annotated, lanes = run_inference(frame_bgr)
        col2.image(annotated, caption=f"Detected {len(lanes)} lanes")

else:   # Video
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded:
        # Save to temp file so cv2.VideoCapture can open it
        tmp_in = "/tmp/streamlit_in.mp4"
        with open(tmp_in, "wb") as f:
            f.write(uploaded.read())
        
        if st.button("Process video"):
            cap = cv2.VideoCapture(tmp_in)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            
            tmp_out = "/tmp/streamlit_out.mp4"
            writer = None
            
            preview = st.empty()
            progress = st.progress(0)
            
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok: break
                annotated, _ = run_inference(frame)
                annotated_bgr = cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
                if writer is None:
                    h, w = annotated_bgr.shape[:2]
                    writer = cv2.VideoWriter(tmp_out,
                                             cv2.VideoWriter_fourcc(*'mp4v'),
                                             fps, (w, h))
                writer.write(annotated_bgr)
                
                # Update live preview every 5 frames
                if i % 5 == 0:
                    preview.image(annotated, caption=f"Frame {i}/{total}")
                progress.progress((i+1) / total)
                i += 1
            
            cap.release()
            if writer: writer.release()
            
            st.success(f"Processed {i} frames")
            st.video(tmp_out)            # Final playback
            
            with open(tmp_out, "rb") as f:
                st.download_button("Download annotated video", f, "annotated.mp4", "video/mp4")