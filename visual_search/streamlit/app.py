import json
import os
import pickle
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
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
TOP_K_VALUES = (5, 10, 15)
MAX_CANDIDATES = max(TOP_K_VALUES)


st.set_page_config(page_title="Visual Product Search", layout="wide")
st.title("Visual Product Search")
st.caption("Upload -> YOLOS detect/crop -> Confirm crop -> CLIP retrieval from fused HNSW -> BLIP-2 ITM rerank")


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


def load_finetuned_clip_model(checkpoint_path: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=DEVICE,
    )
    state = torch.load(checkpoint_path, map_location=DEVICE)
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
def load_clip(checkpoint_path: str):
    return load_finetuned_clip_model(checkpoint_path)


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

    csv_path = os.path.join(index_dir, "gallery_index_blip.csv")
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
    idx_path = os.path.join(index_dir, "gallery_fused_hnsw.bin")
    if not os.path.exists(idx_path):
        raise FileNotFoundError(
            f"Missing HNSW index: {idx_path}. Run build_index.py once to create it."
        )

    config = load_config(index_dir)
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
        logits = model(**inputs)[0]
    return F.softmax(logits.float(), dim=-1)[:, 1].cpu().numpy()


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


def hnsw_retrieve(index, query_emb: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    labels, distances = index.knn_query(query_emb, k=k)
    # cosine distance -> similarity
    sims = 1.0 - distances[0]
    return labels[0], sims


st.sidebar.title("Settings")
index_dir = st.sidebar.text_input("Index Directory", "./index")
gallery_dir = st.sidebar.text_input("Gallery Images Directory", "./data/split_images/gallery")
original_dir = st.sidebar.text_input("Original Images Directory (optional)", "")
checkpoint_path = st.sidebar.text_input("Finetuned CLIP checkpoint", "./clip_finetuned.pt")
det_threshold = st.sidebar.number_input("YOLOS confidence", 0.01, 0.99, 0.30, step=0.05)
st.sidebar.caption("Leave Original Images Directory blank for now; when you share the folder, matching filenames will be loaded from there.")

if not os.path.exists(checkpoint_path):
    st.error(f"Missing finetuned checkpoint: {checkpoint_path}")
    st.stop()

try:
    config = load_config(index_dir)
    gallery_df = load_gallery_df(index_dir)
    hnsw_index = load_hnsw_index(index_dir)
except Exception as exc:
    st.error(str(exc))
    st.stop()

expected_alpha_img = float(config.get("alpha_image", 0.7))
expected_alpha_text = float(config.get("alpha_text", 0.3))
st.sidebar.caption(f"Fused index weights: image={expected_alpha_img:.1f}, text={expected_alpha_text:.1f}")

with st.spinner("Loading YOLOS, finetuned CLIP, and BLIP-2 ITM..."):
    yolo_processor, yolo_model = load_detection_model()
    clip_model, clip_preprocess = load_clip(checkpoint_path)
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

        candidate_rows = gallery_df.iloc[idxs]
        candidate_captions = [row_caption(row) for _, row in candidate_rows.iterrows()]
        itm = itm_scores(itm_processor, itm_model, itm_dtype, query_crop, candidate_captions)

        ranked = []
        for pos, (idx, clip_score, itm_score) in enumerate(zip(idxs, sims, itm), 1):
            ranked.append(
                {
                    "index": int(idx),
                    "clip_score": float(clip_score),
                    "itm_score": float(itm_score),
                    "initial_rank": pos,
                }
            )

        st.session_state.results = ranked

if st.session_state.results:
    img_col = infer_image_column(gallery_df, config)
    if img_col is None:
        st.error("Could not infer image column from gallery CSV.")
        st.stop()

    tabs = st.tabs([f"Top {k}" for k in TOP_K_VALUES])
    for tab, k in zip(tabs, TOP_K_VALUES):
        with tab:
            subset = sorted(st.session_state.results[:k], key=lambda r: r["itm_score"], reverse=True)
            for rank, item in enumerate(subset, 1):
                row = gallery_df.iloc[item["index"]]
                fpath, image_source = resolve_display_image_path(row, img_col, gallery_dir, original_dir)
                caption = row_caption(row)
                original_name = row.get("original_file_name", os.path.basename(str(row[img_col])))
                image_col, detail_col = st.columns([0.25, 0.75])
                with image_col:
                    if os.path.exists(fpath):
                        st.image(fpath, use_container_width=True)
                    else:
                        st.warning("image not found")
                with detail_col:
                    st.write(f"**Final rank #{rank}**")
                    st.write(caption)
                    st.caption(f"File: {original_name}")
                    st.caption(f"Image source: {image_source}")
                    st.caption(f"ITM: {item['itm_score']:.4f}")
                    st.caption(f"HNSW cosine: {item['clip_score']:.4f} | initial #{item['initial_rank']}")
