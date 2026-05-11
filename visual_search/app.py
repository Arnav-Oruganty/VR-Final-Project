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
    .blip-b  { background:#107c10; color:white; }
    .itm-b   { background:#8764b8; color:white; }
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
def load_blip_itm_model():
    """Load BLIP ITM model (~450 MB, cached after first download)."""
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc   = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model  = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    model.to(device).eval()
    return proc, model, device


@st.cache_resource
def load_index(index_dir):
    try:
        clip_embs = np.load(os.path.join(index_dir, 'gallery_embs.npy'))

        # BLIP text embeddings (pre-computed from captions)
        blip_embs = None
        blip_embs_path = os.path.join(index_dir, 'gallery_text_embs.npy')
        if os.path.exists(blip_embs_path):
            blip_embs = np.load(blip_embs_path)

        # Gallery metadata — prefer the one with captions
        blip_csv  = os.path.join(index_dir, 'gallery_index_blip.csv')
        plain_csv = os.path.join(index_dir, 'gallery_index.csv')
        df = pd.read_csv(blip_csv if os.path.exists(blip_csv) else plain_csv)

        with open(os.path.join(index_dir, 'config.json')) as f:
            cfg = json.load(f)

        return clip_embs, blip_embs, df, cfg
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None, None, None, None


# ── BLIP ITM re-ranker (neural) ───────────────────────────────────────────────
def blip_itm_rerank_neural(query_img, candidate_indices, gallery_df,
                            proc, blip_model, blip_device, batch_size=16):
    """
    Run BLIP ITM head to score (query_image, gallery_caption) pairs.
    Returns sorted_indices and ITM match probabilities.
    """
    captions = [
        str(gallery_df.iloc[i].get('clean_caption', 'a clothing item'))
        for i in candidate_indices
    ]
    all_probs = []
    for s in range(0, len(captions), batch_size):
        batch_caps  = captions[s:s + batch_size]
        batch_imgs  = [query_img] * len(batch_caps)
        inputs = proc(
            images=batch_imgs, text=batch_caps,
            return_tensors="pt", padding=True,
            truncation=True, max_length=64
        ).to(blip_device)
        with torch.no_grad():
            out = blip_model(**inputs, use_itm_head=True)
        probs = F.softmax(out.itm_score, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())

    paired = sorted(zip(candidate_indices, all_probs),
                    key=lambda x: x[1], reverse=True)
    return [p[0] for p in paired], [p[1] for p in paired]


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
INDEX_DIR = st.sidebar.text_input("Index Directory", "./index")
IMG_DIR   = st.sidebar.text_input(
    "Images Directory", "./data/archive/cropped_images"
)
top_k = st.sidebar.slider("Final Top-K", 1, 20, 10)

st.sidebar.markdown("---")
st.sidebar.subheader("🔬 Re-ranking Mode")
rerank_mode = st.sidebar.radio(
    "After CLIP retrieval:",
    ["None (CLIP only)", "BLIP Text Similarity", "BLIP ITM (neural)"],
    index=0,
    help=(
        "**BLIP Text Sim**: cosine similarity between query-image CLIP embedding "
        "and pre-computed gallery-caption CLIP-text embeddings. Fast.\n\n"
        "**BLIP ITM (neural)**: runs the BLIP ITM model head on each candidate. "
        "Most accurate but slower (~10–20s for 50 candidates on CPU)."
    )
)
clip_cands = st.sidebar.slider(
    "CLIP candidates to re-rank", 20, 100, 50,
    disabled=(rerank_mode == "None (CLIP only)")
)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛍️ Visual Product Search")
st.markdown("**DeepFashion In-Shop Retrieval · Condition A + B**")

badge_str = '<span class="badge clip-b">Stage 1 · CLIP Vision</span>'
if rerank_mode == "BLIP Text Similarity":
    badge_str += ' → <span class="badge blip-b">Stage 2 · BLIP Caption Re-rank</span>'
elif rerank_mode == "BLIP ITM (neural)":
    badge_str += ' → <span class="badge itm-b">Stage 2 · BLIP ITM Neural Re-rank</span>'
st.markdown(badge_str, unsafe_allow_html=True)
st.markdown("")

# ── Load models & index ───────────────────────────────────────────────────────
with st.spinner("Loading CLIP + YOLO…"):
    yolo_model, clip_model, clip_preprocess, clip_tokenizer, clip_device = load_clip()

if not os.path.exists(INDEX_DIR):
    st.warning(f"⚠️ Index directory '{INDEX_DIR}' not found.")
    st.stop()

gallery_embs, gallery_text_embs, gallery_df, config = load_index(INDEX_DIR)
if gallery_embs is None:
    st.stop()

has_captions  = 'clean_caption' in gallery_df.columns
has_text_embs = gallery_text_embs is not None

st.sidebar.success(f"✅ {len(gallery_embs):,} CLIP embeddings")
if has_text_embs:
    st.sidebar.success(f"✅ {len(gallery_text_embs):,} BLIP text embeddings")
else:
    if rerank_mode != "None (CLIP only)":
        st.sidebar.warning("⚠️ gallery_text_embs.npy not found — BLIP text similarity unavailable")

# Load BLIP ITM model only if needed
blip_proc = blip_itm = blip_dev = None
if rerank_mode == "BLIP ITM (neural)":
    with st.spinner("Loading BLIP ITM model (first time ~1 min)…"):
        blip_proc, blip_itm, blip_dev = load_blip_itm_model()
    st.sidebar.success("✅ BLIP ITM model ready")

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

        # ── Stage 1: CLIP visual retrieval ────────────────────────────────────
        with st.spinner("Stage 1 · CLIP visual retrieval…"):
            img_t = clip_preprocess(query_crop).unsqueeze(0).to(clip_device)
            with torch.no_grad():
                q_img_emb = clip_model.encode_image(img_t)
                q_img_emb = F.normalize(q_img_emb.float(), dim=-1).cpu().numpy()

            clip_sims = np.dot(gallery_embs, q_img_emb.T).flatten()
            n_retrieve = clip_cands if rerank_mode != "None (CLIP only)" else top_k
            clip_top   = np.argsort(clip_sims)[::-1][:n_retrieve].tolist()

        # ── Stage 2: Re-ranking ───────────────────────────────────────────────
        if rerank_mode == "BLIP Text Similarity" and has_text_embs:
            # Cross-modal: CLIP image emb vs pre-computed CLIP text embs of captions
            with st.spinner("Stage 2 · BLIP caption text similarity re-ranking…"):
                candidate_text_embs = gallery_text_embs[clip_top]  # (N, 768)
                text_sims = np.dot(candidate_text_embs, q_img_emb.T).flatten()
                reranked_order = np.argsort(text_sims)[::-1]
                top_indices    = [clip_top[i] for i in reranked_order[:top_k]]
                display_scores = text_sims[reranked_order[:top_k]].tolist()
                score_label    = "BLIP text sim"

        elif rerank_mode == "BLIP ITM (neural)" and blip_itm is not None:
            with st.spinner(f"Stage 2 · BLIP ITM neural re-ranking {n_retrieve} candidates…"):
                top_indices, display_scores = blip_itm_rerank_neural(
                    query_crop, clip_top, gallery_df,
                    blip_proc, blip_itm, blip_dev
                )
                top_indices    = top_indices[:top_k]
                display_scores = display_scores[:top_k]
                score_label    = "ITM prob"

        else:
            top_indices    = clip_top[:top_k]
            display_scores = [float(clip_sims[i]) for i in top_indices]
            score_label    = "CLIP cos"

        # ── Results grid ──────────────────────────────────────────────────────
        st.markdown("---")
        st.header(f"Top {top_k} Results  ·  {score_label}")

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
                score    = display_scores[pos]
                row_data = gallery_df.iloc[idx]
                fname    = row_data[img_col]
                fpath    = fname if os.path.isabs(str(fname)) else os.path.join(img_dir, fname)

                with cols[c]:
                    if os.path.exists(fpath):
                        st.image(fpath, use_container_width=True)
                    else:
                        st.warning("🖼️ not found")
                    st.write(f"**{score_label}: {score:.4f}**")
                    cap = row_data.get('clean_caption', '')
                    if cap:
                        st.caption(f"*{cap}*")
                    st.caption(f"ID: {row_data.get('item_id','N/A')}")

st.markdown("---")
st.caption("DeepFashion In-Shop Retrieval — Condition A (CLIP) + Condition B (BLIP ITM Re-ranking)")
