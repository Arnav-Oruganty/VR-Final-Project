import streamlit as st
import torch
import torch.nn.functional as F
import open_clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os
import json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    .stButton>button {
        width:100%; border-radius:8px; height:3em;
        background:linear-gradient(135deg,#ff4b4b,#d63031);
        color:white; font-weight:bold; border:none;
    }
    .badge {
        display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:.8em; font-weight:bold; margin:2px;
    }
    .clip-b  { background:#0078d4; color:white; }
</style>
""", unsafe_allow_html=True)


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo = YOLO('yolov8n.pt')
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    tokenizer = open_clip.get_tokenizer('ViT-L-14')
    model.to(device).eval()
    return yolo, model, preprocess, tokenizer, device


@st.cache_resource
def load_index(index_dir):
    try:
        clip_embs = np.load(os.path.join(index_dir, 'gallery_embs.npy'))

        plain_csv = os.path.join(index_dir, 'gallery_index.csv')
        blip_csv  = os.path.join(index_dir, 'gallery_index_blip.csv')
        df = pd.read_csv(blip_csv if os.path.exists(blip_csv) else plain_csv)

        with open(os.path.join(index_dir, 'config.json')) as f:
            cfg = json.load(f)

        return clip_embs, df, cfg
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None, None, None


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
INDEX_DIR = st.sidebar.text_input("Index Directory", "./index")
IMG_DIR   = st.sidebar.text_input(
    "Images Directory", "./data/archive/cropped_images"
)
top_k = st.sidebar.slider("Top-K results", 1, 20, 10)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛍️ Visual Product Search")
st.markdown("**DeepFashion In-Shop Retrieval · CLIP Vision Search**")
st.markdown('<span class="badge clip-b">CLIP Vision Search</span>', unsafe_allow_html=True)
st.markdown("")

# ── Load models & index ───────────────────────────────────────────────────────
with st.spinner("Loading CLIP + YOLO…"):
    yolo_model, clip_model, clip_preprocess, clip_tokenizer, clip_device = load_clip()

if not os.path.exists(INDEX_DIR):
    st.warning(f"⚠️ Index directory '{INDEX_DIR}' not found.")
    st.stop()

gallery_embs, gallery_df, config = load_index(INDEX_DIR)
if gallery_embs is None:
    st.stop()

st.sidebar.success(f"✅ {len(gallery_embs):,} CLIP embeddings loaded")

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a clothing image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    fb       = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_cv   = cv2.imdecode(fb, 1)
    img_rgb  = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img  = Image.fromarray(img_rgb)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Query Image")
        st.image(pil_img, use_container_width=True)

    res   = yolo_model(img_rgb, verbose=False)[0]
    boxes = res.boxes.xyxy.cpu().numpy()
    query_crop = pil_img

    with c2:
        st.subheader("YOLO Detection")
        if len(boxes) > 0:
            box = boxes[0].astype(int)
            h, w = img_rgb.shape[:2]
            box  = [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])]
            query_crop = pil_img.crop(box)
            st.image(query_crop, caption="Detected crop (used for search)", use_container_width=True)
            if not st.checkbox("Use crop for search", value=True):
                query_crop = pil_img
        else:
            st.info("No product detected — using full image.")
            st.image(pil_img, use_container_width=True)

    if st.button("🔍 Search"):

        # ── CLIP visual retrieval ──────────────────────────────────────────────
        with st.spinner("CLIP visual retrieval…"):
            img_t = clip_preprocess(query_crop).unsqueeze(0).to(clip_device)
            with torch.no_grad():
                q_img_emb = clip_model.encode_image(img_t)
                q_img_emb = F.normalize(q_img_emb.float(), dim=-1).cpu().numpy()

            clip_sims  = np.dot(gallery_embs, q_img_emb.T).flatten()
            top_indices = np.argsort(clip_sims)[::-1][:top_k].tolist()
            top_scores  = [float(clip_sims[i]) for i in top_indices]

        # ── Results grid ──────────────────────────────────────────────────────
        st.markdown("---")
        st.header(f"Top {top_k} Results  ·  CLIP cosine similarity")

        img_dir = IMG_DIR if IMG_DIR else config.get('img_dir', '')
        img_col = config.get('img_col', 'resolved_filename')

        cols_per_row = 5
        for row in range((top_k + cols_per_row - 1) // cols_per_row):
            cols = st.columns(cols_per_row)
            for c in range(cols_per_row):
                pos = row * cols_per_row + c
                if pos >= len(top_indices):
                    break
                idx      = top_indices[pos]
                score    = top_scores[pos]
                row_data = gallery_df.iloc[idx]
                fname    = row_data[img_col]
                fpath    = fname if os.path.isabs(str(fname)) else os.path.join(img_dir, fname)

                with cols[c]:
                    if os.path.exists(fpath):
                        st.image(fpath, use_container_width=True)
                    else:
                        st.warning("🖼️ not found")
                    st.write(f"**CLIP cos: {score:.4f}**")
                    cap = row_data.get('clean_caption', '')
                    if cap:
                        st.caption(f"*{cap}*")
                    st.caption(f"ID: {row_data.get('item_id', 'N/A')}")

st.markdown("---")
st.caption("DeepFashion In-Shop Retrieval — CLIP Vision Search")
