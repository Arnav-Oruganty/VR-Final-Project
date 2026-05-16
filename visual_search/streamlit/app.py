import base64
import html
import io
import json
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import hnswlib
import numpy as np
import open_clip
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    AutoProcessor,
    Blip2ForImageTextRetrieval,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
YOLOS_MODEL_ID = "valentinafevu/yolos-fashionpedia"
BLIP2_ITM_MODEL_ID = "Salesforce/blip2-itm-vit-g"
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
TOP_K_VALUES = (5, 10, 15)
MAX_CANDIDATES = max(TOP_K_VALUES)
PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_DIR = PROJECT_DIR / "index"


st.set_page_config(page_title="Visual Product Search", layout="wide")
st.title("Visual Product Search")
st.caption("Upload -> YOLOS detect/crop -> Confirm crop -> CLIP retrieval from fused HNSW -> BLIP-2 ITM rerank")

st.markdown(
    """
    <style>
    .result-card {
        display: grid;
        grid-template-columns: 184px minmax(0, 1fr);
        gap: 18px;
        align-items: stretch;
        padding: 14px;
        margin: 0 0 14px 0;
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 8px;
        background: rgba(15, 23, 42, 0.28);
    }
    .result-image-frame {
        width: 184px;
        height: 224px;
        border-radius: 6px;
        background: #f8fafc;
        border: 1px solid rgba(148, 163, 184, 0.2);
        overflow: hidden;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .result-image-frame img {
        width: 100%;
        height: 100%;
        object-fit: contain;
        display: block;
    }
    .result-title {
        font-size: 1rem;
        font-weight: 700;
        line-height: 1.2;
        margin-bottom: 6px;
    }
    .result-caption {
        font-size: 0.88rem;
        line-height: 1.35;
        margin: 8px 0 10px 0;
        color: rgba(226, 232, 240, 0.92);
    }
    .result-meta {
        font-size: 0.72rem;
        line-height: 1.35;
        color: rgba(148, 163, 184, 0.95);
        word-break: break-word;
    }
    .score-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(110px, 1fr));
        gap: 8px;
        margin-top: 12px;
        max-width: 340px;
    }
    .score-box {
        border-top: 1px solid rgba(148, 163, 184, 0.22);
        padding-top: 7px;
    }
    .score-label {
        display: block;
        font-size: 0.68rem;
        color: rgba(148, 163, 184, 0.95);
        margin-bottom: 2px;
    }
    .score-value {
        display: block;
        font-size: 1.02rem;
        font-weight: 700;
        color: #f8fafc;
        font-variant-numeric: tabular-nums;
    }
    .missing-image {
        color: #64748b;
        font-size: 0.78rem;
    }
    @media (max-width: 720px) {
        .result-card {
            grid-template-columns: 128px minmax(0, 1fr);
            gap: 12px;
            padding: 10px;
        }
        .result-image-frame {
            width: 128px;
            height: 168px;
        }
        .score-grid {
            grid-template-columns: 1fr;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_yolos():
    processor = AutoImageProcessor.from_pretrained("valentinafevu/yolos-fashionpedia")
    model = AutoModelForObjectDetection.from_pretrained(
        "valentinafevu/yolos-fashionpedia"
    ).to(DEVICE)
    model.eval()
    return processor, model


@st.cache_resource
def load_detection_model():
    return load_yolos()


def load_finetuned_clip_model(checkpoint_path: str, model_name: str, pretrained: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name,
        pretrained=pretrained,
        device=DEVICE,
    )
    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    if isinstance(state, dict):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()
    return model, preprocess


@st.cache_resource
def load_clip(checkpoint_path: str, model_name: str, pretrained: str):
    return load_finetuned_clip_model(checkpoint_path, model_name, pretrained)


@st.cache_resource
def load_itm_model():
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    processor = AutoProcessor.from_pretrained(BLIP2_ITM_MODEL_ID)
    model = Blip2ForImageTextRetrieval.from_pretrained(
        BLIP2_ITM_MODEL_ID,
        torch_dtype=dtype,
    ).to(DEVICE)
    model.eval()
    return processor, model, dtype


@st.cache_data
def load_config(index_dir: str):
    config_path = os.path.join(index_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Missing config: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def load_gallery_df(index_dir: str):
    config = load_config(index_dir)
    metadata_file = config.get("metadata_file", "gallery_metadata.pkl")
    pickle_path = os.path.join(index_dir, metadata_file)
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            payload = pickle.load(f)
        if isinstance(payload, dict) and "metadata" in payload:
            return add_derived_metadata(pd.DataFrame(payload["metadata"]))
        if isinstance(payload, pd.DataFrame):
            return add_derived_metadata(payload)

    csv_path = config.get("captions_csv") or os.path.join(index_dir, "gallery_index_blip.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(index_dir, "gallery_index.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Missing gallery_index_blip.csv / gallery_index.csv")
    return add_derived_metadata(pd.read_csv(csv_path))


def add_derived_metadata(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "image_path" in df.columns and "original_file_name" not in df.columns:
        df["original_file_name"] = df["image_path"].map(lambda value: os.path.basename(str(value)))
    return df


@st.cache_resource
def load_hnsw_index(index_dir: str):
    config = load_config(index_dir)
    idx_path = os.path.join(index_dir, config.get("hnsw_index_file", "gallery_fused_hnsw.bin"))
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Missing HNSW index: {idx_path}. Run build_index.py once to create it."
        )

    dim = int(config.get("fused_dim", config.get("embedding_dim", 768)))
    num_items = int(config.get("num_indexed", 0))
    if num_items <= 0:
        raise ValueError("config.json has invalid num_indexed; rebuild index.")

    index = hnswlib.Index(space="cosine", dim=dim)
    index.load_index(idx_path, max_elements=num_items)
    return index


def detect_fashion_objects(processor, model, image: Image.Image, threshold: float):
    inputs = processor(images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs)
    result = processor.post_process_object_detection(
        outputs,
        threshold=threshold,
        target_sizes=torch.tensor([image.size[::-1]], device=DEVICE),
    )[0]

    detections = []
    width, height = image.size
    for score, label_id, box in zip(result["scores"], result["labels"], result["boxes"]):
        x1, y1, x2, y2 = box.detach().cpu().numpy().tolist()
        x1 = int(max(0, min(width - 1, round(x1))))
        y1 = int(max(0, min(height - 1, round(y1))))
        x2 = int(max(1, min(width, round(x2))))
        y2 = int(max(1, min(height, round(y2))))
        if x2 <= x1 or y2 <= y1:
            continue
        label_id = int(label_id.detach().cpu().item())
        detections.append(
            {
                "label": model.config.id2label.get(label_id, str(label_id)),
                "score": float(score.detach().cpu().item()),
                "box": (x1, y1, x2, y2),
            }
        )

    detections.sort(key=lambda d: d["score"], reverse=True)
    for idx, det in enumerate(detections, 1):
        det["idx"] = idx
    return detections


def draw_detections(image: Image.Image, detections, selected_idx: int):
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    for det in detections:
        color = (0, 180, 120) if det["idx"] == selected_idx else (64, 120, 220)
        x1, y1, x2, y2 = det["box"]
        text = f'{det["idx"]}. {det["label"]} {det["score"]:.2f}'
        draw.rectangle((x1, y1, x2, y2), outline=color, width=4 if det["idx"] == selected_idx else 2)
        draw.text((x1 + 3, max(0, y1 - 12)), text, fill=color, font=font)
    return annotated


def encode_image(model, preprocess, image: Image.Image) -> np.ndarray:
    image_t = preprocess(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model.encode_image(image_t)
    emb = F.normalize(emb.float(), dim=-1).cpu().numpy().astype("float32")
    return emb


def itm_scores(processor, model, dtype, query_image: Image.Image, captions: List[str]) -> np.ndarray:
    clean = [c if isinstance(c, str) and c.strip() else "a clothing item" for c in captions]
    inputs = processor(
        images=[query_image] * len(clean),
        text=clean,
        return_tensors="pt",
        padding=True,
    )
    inputs = {
        key: value.to(DEVICE, dtype=dtype) if torch.is_floating_point(value) else value.to(DEVICE)
        for key, value in inputs.items()
    }
    with torch.no_grad():
        outputs = model(**inputs, use_image_text_matching_head=True)
    return F.softmax(outputs.logits_per_image.float(), dim=1)[:, 1].cpu().numpy()


def row_caption(row) -> str:
    for col in ("caption", "clean_caption", "blip_caption"):
        if col in row and pd.notna(row[col]) and str(row[col]).strip():
            return str(row[col])
    return "a clothing item"


def infer_image_column(df: pd.DataFrame, config: dict):
    configured = config.get("img_col")
    if configured in df.columns:
        return configured
    for col in ("filename", "cropped_image_path", "image_path", "file_name", "path"):
        if col in df.columns:
            return col
    return None


def resolve_gallery_image_path(value: str, img_dir: str):
    value = str(value)
    if os.path.isabs(value) and os.path.exists(value):
        return value
    joined = os.path.join(img_dir, value)
    if os.path.exists(joined):
        return joined
    return os.path.join(img_dir, os.path.basename(value))


def resolve_display_image_path(row, img_col: str, gallery_dir: str, original_dir: str):
    source_value = str(row[img_col])
    filename = str(row.get("original_file_name", os.path.basename(source_value)))
    if original_dir.strip():
        candidate = os.path.join(original_dir.strip(), filename)
        if os.path.exists(candidate):
            return candidate, "original"
    return resolve_gallery_image_path(source_value, gallery_dir), "gallery"


def image_data_uri(path: str) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=88)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def render_result_card(item: dict, caption: str, image_path: str, image_source: str, original_name: str):
    if os.path.exists(image_path):
        image_html = f'<img src="{image_data_uri(image_path)}" alt="{html.escape(str(original_name))}">'
    else:
        image_html = '<span class="missing-image">Image not found</span>'

    st.markdown(
        f"""
        <div class="result-card">
            <div class="result-image-frame">{image_html}</div>
            <div>
                <div class="result-title">Rank {item["final_rank"]}</div>
                <div class="result-meta">Item ID: {item["index"]} | HNSW rank before BLIP-2: {item["pre_rerank_rank"]}</div>
                <div class="result-meta">File: {html.escape(str(original_name))}</div>
                <div class="result-meta">Image source: {html.escape(str(image_source))}</div>
                <div class="result-caption">{html.escape(str(caption))}</div>
                <div class="score-grid">
                    <div class="score-box">
                        <span class="score-label">CLIP/HNSW similarity</span>
                        <span class="score-value">{item["clip_score"]:.4f}</span>
                    </div>
                    <div class="score-box">
                        <span class="score-label">BLIP-2 match score</span>
                        <span class="score-value">{item["itm_score"]:.4f}</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def hnsw_retrieve(index, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    labels, distances = index.knn_query(query_emb, k=k)
    # cosine distance -> similarity
    sims = 1.0 - distances[0]
    return labels[0], sims


st.sidebar.title("Settings")
index_dir = st.sidebar.text_input("Index Directory", str(DEFAULT_INDEX_DIR))
det_threshold = st.sidebar.number_input("YOLOS confidence", 0.01, 0.99, 0.30, step=0.05)

try:
    config = load_config(index_dir)
    gallery_df = load_gallery_df(index_dir)
    hnsw_index = load_hnsw_index(index_dir)
except Exception as exc:
    st.error(str(exc))
    st.stop()

gallery_dir = st.sidebar.text_input(
    "Gallery Images Directory",
    config.get("gallery_dir", "./data/split_images/gallery"),
)
original_dir = st.sidebar.text_input("Original Images Directory (optional)", "")
checkpoint_path = st.sidebar.text_input(
    "Finetuned CLIP checkpoint",
    config.get("checkpoint_path", "./clip-model/clip_finetuned.pt"),
)
st.sidebar.caption("Leave Original Images Directory blank for now; when you share the folder, matching filenames will be loaded from there.")

if not os.path.exists(checkpoint_path):
    st.error(f"Missing finetuned checkpoint: {checkpoint_path}")
    st.stop()

expected_alpha_img = float(config.get("alpha_image", 0.8))
expected_alpha_text = float(config.get("alpha_text", 0.2))
clip_model_name = config.get("model", CLIP_MODEL_NAME)
clip_pretrained = config.get("pretrained", CLIP_PRETRAINED)
st.sidebar.caption(f"Fused index weights: image={expected_alpha_img:.1f}, text={expected_alpha_text:.1f}")
st.sidebar.caption(f"CLIP model: {clip_model_name}")

with st.spinner("Loading YOLOS, finetuned CLIP, and BLIP-2 ITM..."):
    yolo_processor, yolo_model = load_detection_model()
    clip_model, clip_preprocess = load_clip(checkpoint_path, clip_model_name, clip_pretrained)
    itm_processor, itm_model, itm_dtype = load_itm_model()

for key, default in {
    "upload_id": None,
    "image": None,
    "detections": [],
    "selected_idx": None,
    "crop_confirmed": False,
    "results": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

uploaded = st.file_uploader("Upload input image", type=["jpg", "jpeg", "png"])
if uploaded is None:
    st.info("Upload an image to begin.")
    st.stop()

upload_id = f"{uploaded.name}:{uploaded.size}:{det_threshold:.2f}"
if st.session_state.upload_id != upload_id:
    image = Image.open(uploaded).convert("RGB")
    with st.spinner("Running YOLOS detection..."):
        detections = detect_fashion_objects(yolo_processor, yolo_model, image, det_threshold)
    st.session_state.upload_id = upload_id
    st.session_state.image = image
    st.session_state.detections = detections
    st.session_state.selected_idx = detections[0]["idx"] if detections else None
    st.session_state.crop_confirmed = False
    st.session_state.results = None

image = st.session_state.image
detections = st.session_state.detections
if not detections:
    st.error("No YOLOS garment detections. Lower confidence or try another image.")
    st.stop()

left, right = st.columns([1.4, 1])
with left:
    st.subheader("Detected objects")
    st.image(draw_detections(image, detections, st.session_state.selected_idx), use_container_width=True)

with right:
    labels = [f'{d["idx"]}. {d["label"]} (conf {d["score"]:.3f})' for d in detections]
    selected_label = st.radio("Choose crop", labels)
    selected_idx = int(selected_label.split(".", 1)[0])
    if selected_idx != st.session_state.selected_idx:
        st.session_state.selected_idx = selected_idx
        st.session_state.crop_confirmed = False
        st.session_state.results = None
        st.rerun()

selected = next(d for d in detections if d["idx"] == st.session_state.selected_idx)
query_crop = image.crop(selected["box"])

c1, c2 = st.columns([1, 1])
with c1:
    st.image(query_crop, caption=f'{selected["label"]} | conf {selected["score"]:.3f}', width=280)
with c2:
    confirm_col, recrop_col = st.columns(2)
    if confirm_col.button("Confirm crop", type="primary"):
        st.session_state.crop_confirmed = True
        st.session_state.results = None
        st.rerun()
    if recrop_col.button("Re-crop"):
        st.session_state.crop_confirmed = False
        st.session_state.results = None
        st.rerun()

if not st.session_state.crop_confirmed:
    st.info("Confirm crop before retrieval.")
    st.stop()

if st.button("Retrieve Top K and Re-rank", type="primary"):
    with st.spinner("Encoding crop, retrieving with HNSW cosine search, and ITM re-ranking..."):
        query_emb = encode_image(clip_model, clip_preprocess, query_crop)
        idxs, sims = hnsw_retrieve(hnsw_index, query_emb, MAX_CANDIDATES)
        pre_rerank_rank = {int(idx): rank for rank, idx in enumerate(idxs, 1)}

        candidate_rows = gallery_df.iloc[idxs]
        candidate_captions = [row_caption(row) for _, row in candidate_rows.iterrows()]
        itm = itm_scores(itm_processor, itm_model, itm_dtype, query_crop, candidate_captions)

        candidates = []
        for idx, clip_score, itm_score in zip(idxs, sims, itm):
            candidates.append(
                {
                    "index": int(idx),
                    "clip_score": float(clip_score),
                    "itm_score": float(itm_score),
                    "pre_rerank_rank": pre_rerank_rank[int(idx)],
                }
            )
        ranked = sorted(candidates, key=lambda r: r["itm_score"], reverse=True)
        for final_rank, item in enumerate(ranked, 1):
            item["final_rank"] = final_rank

        st.session_state.results = ranked

if st.session_state.results:
    img_col = infer_image_column(gallery_df, config)
    if img_col is None:
        st.error("Could not infer image column from gallery CSV.")
        st.stop()

    tabs = st.tabs([f"Top {k}" for k in TOP_K_VALUES])
    for tab, k in zip(tabs, TOP_K_VALUES):
        with tab:
            subset = st.session_state.results[:k]
            for item in subset:
                row = gallery_df.iloc[item["index"]]
                fpath, image_source = resolve_display_image_path(row, img_col, gallery_dir, original_dir)
                caption = row_caption(row)
                original_name = row.get("original_file_name", os.path.basename(str(row[img_col])))
                render_result_card(item, caption, fpath, image_source, original_name)
