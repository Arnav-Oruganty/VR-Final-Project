import streamlit as st
import torch
import torch.nn.functional as F
import open_clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
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
    body { background-color: #0e0e0e; }
    .stButton>button {
        width:100%; border-radius:8px; height:3em;
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        color: white; font-weight: bold; border: 1px solid #444;
        transition: all 0.2s;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0f3460, #533483);
        border-color: #7c6af7;
    }
    .badge {
        display:inline-block; padding:3px 10px; border-radius:12px;
        font-size:.8em; font-weight:bold; margin:2px;
    }
    .clip-b  { background:#0078d4; color:white; }
    .step-pill {
        display:inline-block;
        background: #7c6af7;
        color: white;
        font-size: 0.75em;
        font-weight: bold;
        padding: 3px 12px;
        border-radius: 20px;
        margin-bottom: 8px;
    }
    .region-card {
        border: 2px solid #333;
        border-radius: 12px;
        padding: 10px;
        text-align: center;
        background: #1a1a2e;
        transition: border-color 0.2s;
    }
    .region-card:hover { border-color: #7c6af7; }
    .region-card.selected { border-color: #00c896; }
    .conf-badge {
        display:inline-block;
        padding: 2px 8px;
        border-radius: 10px;
        font-size: 0.75em;
        font-weight: bold;
        background: #1e3a2f;
        color: #00c896;
        border: 1px solid #00c896;
        margin-top: 4px;
    }
    .search-btn>button {
        background: linear-gradient(135deg, #ff4b4b, #d63031) !important;
        border: none !important;
        font-size: 1.1em !important;
    }
</style>
""", unsafe_allow_html=True)


# ── YOLO Fashion Region Detector ──────────────────────────────────────────────
REGION_CONFIG = {
    "upper": {
        "label": "Upper Body",
        "icon": "👕",
        "desc": "Tops, shirts, jackets, hoodies",
        "color": (59, 130, 246),    # blue
        "color_hex": "#3B82F6",
        # YOLO class IDs to treat as upper body
        "yolo_classes": {0},        # 'person' fallback handled separately
    },
    "lower": {
        "label": "Lower Body",
        "icon": "👖",
        "desc": "Trousers, skirts, shorts",
        "color": (34, 197, 94),     # green
        "color_hex": "#22C55E",
    },
    "full": {
        "label": "Full Body / Outfit",
        "icon": "🧍",
        "desc": "Dresses, jumpsuits, full look",
        "color": (244, 63, 94),     # pink
        "color_hex": "#F43F5E",
    },
}

# Proportion of person bbox used to derive upper/lower/full crops
SPLIT_RATIO = 0.52   # top 52% = upper, bottom 50% = lower


def derive_garment_regions(person_box, img_w, img_h):
    """
    Given a person bounding box (x1,y1,x2,y2), derive three garment regions.
    Returns dict: region_key -> (x1,y1,x2,y2)
    """
    x1, y1, x2, y2 = person_box
    bw = x2 - x1
    bh = y2 - y1

    # Upper: top portion with slight horizontal padding
    ux1 = max(0, x1 - int(bw * 0.03))
    uy1 = max(0, y1)
    ux2 = min(img_w, x2 + int(bw * 0.03))
    uy2 = min(img_h, y1 + int(bh * SPLIT_RATIO))

    # Lower: bottom portion
    lx1 = max(0, x1 - int(bw * 0.03))
    ly1 = max(0, y1 + int(bh * (SPLIT_RATIO - 0.10)))  # slight overlap
    lx2 = min(img_w, x2 + int(bw * 0.03))
    ly2 = min(img_h, y2)

    # Full body = full person box with small padding
    fx1 = max(0, x1 - int(bw * 0.05))
    fy1 = max(0, y1 - int(bh * 0.02))
    fx2 = min(img_w, x2 + int(bw * 0.05))
    fy2 = min(img_h, y2 + int(bh * 0.02))

    return {
        "upper": (ux1, uy1, ux2, uy2),
        "lower": (lx1, ly1, lx2, ly2),
        "full":  (fx1, fy1, fx2, fy2),
    }


def draw_regions_on_image(pil_img, regions, selected=None):
    """Draw colored region boxes on a copy of pil_img."""
    img = pil_img.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    for key, box in regions.items():
        cfg = REGION_CONFIG[key]
        r, g, b = cfg["color"]
        is_sel = (key == selected)
        alpha_fill = 60 if is_sel else 30
        lw = 4 if is_sel else 2
        draw.rectangle(box, fill=(r, g, b, alpha_fill), outline=(r, g, b, 220), width=lw)

    result = Image.alpha_composite(img, overlay).convert("RGB")
    return result


# ── Loaders ───────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # YOLOv8x — best accuracy in the v8 family
    # Falls back to yolov8n if x is not available (first run downloads it)
    try:
        yolo = YOLO("yolov8x.pt")
    except Exception:
        yolo = YOLO("yolov8n.pt")

    model, _, preprocess = open_clip.create_model_and_transforms(
        'ViT-L-14', pretrained='openai'
    )
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


def run_yolo_person(yolo_model, img_rgb):
    """Run YOLO and return the best person box (class 0)."""
    results = yolo_model(img_rgb, verbose=False, classes=[0])[0]
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return None, 0.0
    confs = boxes.conf.cpu().numpy()
    best_idx = int(np.argmax(confs))
    box = boxes.xyxy[best_idx].cpu().numpy().astype(int).tolist()
    conf = float(confs[best_idx])
    return box, conf


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")
INDEX_DIR = st.sidebar.text_input("Index Directory", "./index")
IMG_DIR   = st.sidebar.text_input("Images Directory", "./data/archive/cropped_images")
top_k     = st.sidebar.slider("Top-K results", 1, 20, 10)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🛍️ Visual Product Search")
st.markdown("**DeepFashion In-Shop Retrieval · CLIP Vision Search**")
st.markdown('<span class="badge clip-b">CLIP Vision Search</span>', unsafe_allow_html=True)
st.markdown("")

# ── Load models & index ───────────────────────────────────────────────────────
with st.spinner("Loading CLIP + YOLOv8x…"):
    yolo_model, clip_model, clip_preprocess, clip_tokenizer, clip_device = load_models()

if not os.path.exists(INDEX_DIR):
    st.warning(f"⚠️ Index directory '{INDEX_DIR}' not found.")
    st.stop()

gallery_embs, gallery_df, config = load_index(INDEX_DIR)
if gallery_embs is None:
    st.stop()

st.sidebar.success(f"✅ {len(gallery_embs):,} CLIP embeddings loaded")

# ── Session state ─────────────────────────────────────────────────────────────
if "step" not in st.session_state:
    st.session_state.step = 1          # 1 = upload, 2 = choose region, 3 = results
if "regions" not in st.session_state:
    st.session_state.regions = None
if "pil_img" not in st.session_state:
    st.session_state.pil_img = None
if "yolo_conf" not in st.session_state:
    st.session_state.yolo_conf = 0.0
if "selected_region" not in st.session_state:
    st.session_state.selected_region = None
if "uploaded_file_id" not in st.session_state:
    st.session_state.uploaded_file_id = None

# ── Step 1 — Upload ───────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a clothing / person image", type=['jpg', 'jpeg', 'png'])

if uploaded is not None:
    # Build a unique ID for the current upload (name + size)
    file_id = f"{uploaded.name}_{uploaded.size}"

    # If the user uploaded a different file, reset all state
    if st.session_state.uploaded_file_id != file_id:
        st.session_state.uploaded_file_id = file_id
        st.session_state.pil_img = None
        st.session_state.regions = None
        st.session_state.yolo_conf = 0.0
        st.session_state.step = 1
        st.session_state.selected_region = None

    fb      = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img_cv  = cv2.imdecode(fb, 1)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    h, w    = img_rgb.shape[:2]

    # Auto-detect on new upload
    if st.session_state.pil_img is None or st.session_state.step == 1:
        with st.spinner("Detecting person with YOLOv8x…"):
            person_box, yolo_conf = run_yolo_person(yolo_model, img_rgb)

        if person_box is not None:
            regions = derive_garment_regions(person_box, w, h)
        else:
            # No person detected → use heuristic splits on full image
            regions = derive_garment_regions([0, 0, w, h], w, h)
            yolo_conf = 0.0

        st.session_state.pil_img   = pil_img
        st.session_state.regions   = regions
        st.session_state.yolo_conf = yolo_conf
        st.session_state.step      = 2
        st.session_state.selected_region = None

    pil_img = st.session_state.pil_img
    regions = st.session_state.regions
    yolo_conf = st.session_state.yolo_conf

    # ── Step 2 — Region selection ──────────────────────────────────────────────
    if st.session_state.step >= 2:
        st.markdown('<div class="step-pill">Step 2 · Choose which garment region to search</div>', unsafe_allow_html=True)

        left_col, right_col = st.columns([1, 2])

        with left_col:
            st.subheader("All detected regions")
            annotated = draw_regions_on_image(pil_img, regions, st.session_state.selected_region)
            st.image(annotated, use_container_width=True)
            conf_color = "#00c896" if yolo_conf > 0.5 else "#facc15"
            st.markdown(
                f'🔵 Upper &nbsp; 🟢 Lower &nbsp; 🩷 Full body &nbsp;|&nbsp; '
                f'YOLO: <span style="color:{conf_color};font-weight:bold">{yolo_conf:.3f} ↑</span>',
                unsafe_allow_html=True
            )

        with right_col:
            st.subheader("Select the region you want to search for:")
            r1, r2, r3 = st.columns(3)

            for col, rkey in zip([r1, r2, r3], ["upper", "lower", "full"]):
                cfg = REGION_CONFIG[rkey]
                box = regions[rkey]
                crop = pil_img.crop(box)

                with col:
                    is_sel = (st.session_state.selected_region == rkey)
                    border_col = "#00c896" if is_sel else cfg["color_hex"]
                    st.markdown(
                        f'<div style="border:2px solid {border_col};border-radius:12px;padding:6px;text-align:center;">',
                        unsafe_allow_html=True
                    )
                    st.image(crop, use_container_width=True)
                    st.markdown(
                        f'<div style="color:{cfg["color_hex"]};font-size:1.4em;margin-top:4px;">{cfg["icon"]}</div>'
                        f'<div style="font-weight:bold;">{cfg["label"]}</div>'
                        f'<div style="font-size:0.8em;color:#aaa;">{cfg["desc"]}</div>'
                        f'<div class="conf-badge">conf {yolo_conf:.3f}</div>',
                        unsafe_allow_html=True
                    )
                    st.markdown('</div>', unsafe_allow_html=True)

                    if st.button(f"Select", key=f"sel_{rkey}"):
                        st.session_state.selected_region = rkey
                        st.session_state.step = 3
                        st.rerun()

        # hint
        if st.session_state.selected_region is None:
            st.info("👆 Click one of the three region buttons above to continue.")

    # ── Step 3 — Search ────────────────────────────────────────────────────────
    if st.session_state.step == 3 and st.session_state.selected_region is not None:
        rkey  = st.session_state.selected_region
        cfg   = REGION_CONFIG[rkey]
        box   = regions[rkey]
        query_crop = pil_img.crop(box)

        st.markdown("---")
        st.markdown(
            f'<div class="step-pill">Step 3 · Searching for {cfg["icon"]} {cfg["label"]}</div>',
            unsafe_allow_html=True
        )

        col_q, col_btn = st.columns([3, 1])
        with col_q:
            st.image(query_crop, caption=f"Query crop: {cfg['label']}", width=220)
        with col_btn:
            st.markdown('<div class="search-btn">', unsafe_allow_html=True)
            do_search = st.button("🔍 Search")
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("← Change region"):
                st.session_state.step = 2
                st.session_state.selected_region = None
                st.rerun()

        if do_search:
            with st.spinner("CLIP visual retrieval…"):
                img_t = clip_preprocess(query_crop).unsqueeze(0).to(clip_device)
                with torch.no_grad():
                    q_img_emb = clip_model.encode_image(img_t)
                    q_img_emb = F.normalize(q_img_emb.float(), dim=-1).cpu().numpy()

                clip_sims   = np.dot(gallery_embs, q_img_emb.T).flatten()
                top_indices = np.argsort(clip_sims)[::-1][:top_k].tolist()
                top_scores  = [float(clip_sims[i]) for i in top_indices]

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
st.caption("DeepFashion In-Shop Retrieval — CLIP Vision Search · YOLOv8x Region Detection")