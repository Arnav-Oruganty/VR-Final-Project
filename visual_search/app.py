import streamlit as st
import torch
import open_clip
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
import os
import json

# Page config
st.set_page_config(
    page_title="Visual Product Search",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
    }
    .result-card {
        padding: 10px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # YOLO for detection
    yolo_model = YOLO('yolov8n.pt')
    # CLIP for embedding
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    clip_model.to(device)
    clip_model.eval()
    return yolo_model, clip_model, clip_preprocess, device

@st.cache_resource
def load_index(index_dir):
    try:
        embs = np.load(os.path.join(index_dir, 'gallery_embs.npy'))
        df = pd.read_csv(os.path.join(index_dir, 'gallery_index.csv'))
        with open(os.path.join(index_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        return embs, df, config
    except Exception as e:
        st.error(f"Error loading index: {e}")
        return None, None, None

# App Layout
st.sidebar.title("⚙️ Settings")
INDEX_DIR = st.sidebar.text_input("Index Directory", "./index")
top_k = st.sidebar.slider("Number of results (Top-K)", 1, 20, 10)

st.title("🛍️ Visual Product Search")
st.markdown("### Condition A: Vision-only Frozen CLIP")
st.info("Upload an image of a clothing item to find similar products in the gallery.")

# Initialize models
with st.spinner("Loading models..."):
    yolo_model, clip_model, clip_preprocess, device = load_models()

# Check for index
if not os.path.exists(INDEX_DIR):
    st.warning(f"⚠️ Index directory '{INDEX_DIR}' not found. Please run `build_index.py` first.")
    st.stop()

gallery_embs, gallery_df, config = load_index(INDEX_DIR)
if gallery_embs is None:
    st.stop()

# Upload
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_cv = cv2.imdecode(file_bytes, 1)
    image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image_rgb)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Query Image")
        st.image(pil_img, use_container_width=True)
        
    # YOLO detection
    results = yolo_model(image_rgb, verbose=False)[0]
    boxes = results.boxes.xyxy.cpu().numpy()
    
    query_crop = pil_img
    
    with col2:
        st.subheader("Detection & Preprocessing")
        if len(boxes) > 0:
            # Take the most confident box
            box = boxes[0].astype(int)
            # Ensure box is within image bounds
            h, w = image_rgb.shape[:2]
            box[0], box[1] = max(0, box[0]), max(0, box[1])
            box[2], box[3] = min(w, box[2]), min(h, box[3])
            
            query_crop = pil_img.crop((box[0], box[1], box[2], box[3]))
            st.image(query_crop, caption="YOLO Detected Crop", use_container_width=True)
            
            use_crop = st.checkbox("Search using YOLO crop", value=True)
            if not use_crop:
                query_crop = pil_img
        else:
            st.warning("No products detected by YOLO. Using full image for search.")
            st.image(pil_img, caption="Full Image (No detection)", use_container_width=True)

    if st.button("🔍 Find Similar Products"):
        with st.spinner("Computing query embedding and searching..."):
            # CLIP Embedding
            image_input = clip_preprocess(query_crop).unsqueeze(0).to(device)
            with torch.no_grad():
                query_emb = clip_model.encode_image(image_input)
                query_emb /= query_emb.norm(dim=-1, keepdim=True)
                query_emb = query_emb.cpu().numpy()
            
            # Cosine similarity
            # gallery_embs: [N, D], query_emb: [1, D]
            similarities = np.dot(gallery_embs, query_emb.T).flatten()
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            st.markdown("---")
            st.header(f"Top {top_k} Similar Products")
            
            # Display grid
            cols_per_row = 5
            rows = (top_k + cols_per_row - 1) // cols_per_row
            
            img_dir = config.get('img_dir', '')
            img_col = config.get('img_col', 'image_path')

            for r in range(rows):
                cols = st.columns(cols_per_row)
                for c in range(cols_per_row):
                    idx_in_top = r * cols_per_row + c
                    if idx_in_top < len(top_indices):
                        idx = top_indices[idx_in_top]
                        res_row = gallery_df.iloc[idx]
                        score = similarities[idx]
                        
                        img_name = res_row[img_col]
                        # Handle both relative and absolute paths
                        if os.path.isabs(img_name):
                            full_img_path = img_name
                        else:
                            full_img_path = os.path.join(img_dir, img_name)
                        
                        with cols[c]:
                            if os.path.exists(full_img_path):
                                st.image(full_img_path, use_container_width=True)
                            else:
                                st.error("Image not found")
                            
                            st.write(f"**Score: {score:.4f}**")
                            # Try to show some item info if available
                            item_id = res_row.get('item_id', res_row.get('id', 'N/A'))
                            st.caption(f"ID: {item_id}")
                            
st.markdown("---")
st.caption("DeepFashion In-Shop Retrieval Visual Search - Condition A (Frozen CLIP)")
