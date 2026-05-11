# 🛍️ Visual Product Search — DeepFashion In-Shop Retrieval

A two-stage visual search system using **Frozen CLIP** (Condition A) and **BLIP ITM Re-ranking** (Condition B) for clothing retrieval on the DeepFashion In-Shop dataset.

---

## Project Structure

```
visual_search/
├── app.py                    # Streamlit search interface
├── build_index.py            # (Optional) Build CLIP gallery index locally
├── requirements.txt          # Python dependencies
├── captions (1).csv          # BLIP2-generated captions (Salesforce/blip2-opt-2.7b)
│
├── index/                    # Pre-built index files (download from Kaggle)
│   ├── gallery_embs.npy          # CLIP ViT-L-14 vision embeddings (12,612 × 768)
│   ├── gallery_text_embs.npy     # CLIP text embeddings of BLIP2 captions (12,612 × 768)
│   ├── gallery_index.csv         # Gallery metadata (image paths, item IDs)
│   ├── gallery_index_blip.csv    # Gallery metadata + BLIP2 captions
│   └── config.json               # Index configuration
│
└── data/
    └── archive/
        ├── cropped_images/       # YOLO-cropped gallery images (~12,612 JPGs)
        ├── cropped_metadata.csv  # Image paths, item IDs, splits
        └── all_metadata.csv      # Full dataset metadata
```

---

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Requirements:** `torch`, `open_clip_torch`, `streamlit`, `ultralytics`, `transformers`, `opencv-python-headless`, `pandas`, `numpy`, `pillow`

---

### 2. Data & Index Files

The index files were generated using Kaggle (GPU) notebooks. You need:

| File | Source | Description |
|------|--------|-------------|
| `cropped_images/` | Kaggle dataset `mithilesh2303/preprocessed-deepfashion` | YOLO-cropped gallery images |
| `gallery_embs.npy` | `a-vision-only-clip.ipynb` → Kaggle output | CLIP vision embeddings |
| `gallery_index.csv` | `a-vision-only-clip.ipynb` → Kaggle output | Gallery metadata |
| `gallery_text_embs.npy` | BLIP index Kaggle notebook → Kaggle output | CLIP text encodings of captions |
| `gallery_index_blip.csv` | BLIP index Kaggle notebook → Kaggle output | Metadata + BLIP2 captions |
| `captions (1).csv` | Provided by team (BLIP2 captioning notebook) | Captions for all 52,712 images |

Place extracted files as shown in the project structure above.

---

### 3. Run the App

```bash
cd visual_search
streamlit run app.py
```

The app opens at **http://localhost:8501**

---

## Using the App

### Sidebar Settings

| Setting | Default | Description |
|---------|---------|-------------|
| **Index Directory** | `./index` | Path to the folder containing `.npy` and `.csv` index files |
| **Images Directory** | `./data/archive/cropped_images` | Path to your local `cropped_images` folder |
| **Final Top-K** | 10 | Number of results to display |
| **Re-ranking Mode** | None | Choose between the 3 retrieval modes (see below) |
| **CLIP candidates to re-rank** | 50 | How many CLIP results BLIP re-ranks |

### 3 Retrieval Modes

| Mode | Speed | How It Works |
|------|-------|--------------|
| **None (CLIP only)** | ⚡ Instant | Cosine similarity between query image embedding and gallery vision embeddings |
| **BLIP Text Similarity** | ⚡ Fast | CLIP retrieves top-N → re-rank by cosine similarity between query image embedding and gallery caption text embeddings (`gallery_text_embs.npy`) |
| **BLIP ITM (neural)** | 🐢 ~10–20s on CPU | CLIP retrieves top-N → `Salesforce/blip-itm-base-coco` scores each (query image, gallery caption) pair and re-ranks by ITM match probability |

### Search Flow

1. **Upload** a clothing image (JPG/PNG)
2. **YOLOv8n** automatically detects and crops the clothing item
3. Toggle the crop on/off using the checkbox
4. Click **🔍 Search**
5. **Stage 1**: CLIP encodes the query image → cosine similarity with gallery
6. **Stage 2** *(if enabled)*: BLIP re-ranks the top candidates
7. Results are displayed in a grid with similarity scores and captions

---

## Rebuilding the Index Locally (Optional)

If you want to rebuild the CLIP gallery index from scratch (takes ~5–10 min on CPU):

```bash
python build_index.py \
    --csv data/archive/all_metadata.csv \
    --img_dir data/archive/cropped_images \
    --out_dir ./index
```

> **Note:** The pre-built index from Kaggle is recommended since it was computed on GPU.

---

## Kaggle Notebooks

| Notebook | Purpose |
|----------|---------|
| `preproc-yolo.ipynb` | YOLOv8-based image cropping + metadata generation |
| `a-vision-only-clip.ipynb` | Frozen CLIP ViT-L-14 gallery indexing + Recall/NDCG/mAP metrics (Condition A) |
| *(BLIP captioning notebook — team)* | BLIP2-opt-2.7b caption generation for all 52,712 images |
| *(BLIP index notebook)* | CLIP text encoding of captions → `gallery_text_embs.npy` |

---

## Models Used

| Model | Use |
|-------|-----|
| `openai/clip-vit-large-patch14` (`ViT-L-14`) | Gallery vision embeddings + query embedding at search time |
| `Salesforce/blip2-opt-2.7b` | Caption generation (offline, done on Kaggle) |
| `Salesforce/blip-itm-base-coco` | BLIP ITM neural re-ranking (loaded on-demand in app) |
| `ultralytics/yolov8n` | Query image clothing detection + cropping |

---

## Condition Summary

| Condition | Method | Index Files Used |
|-----------|--------|-----------------|
| **A** — Vision-only CLIP | CLIP image → CLIP image cosine sim | `gallery_embs.npy` |
| **B** — BLIP Caption Re-ranking | CLIP image → CLIP text cosine sim (fast) or BLIP ITM (neural) | `gallery_text_embs.npy` + `gallery_index_blip.csv` |

---

## Metrics (Condition A — from Kaggle)

Evaluated on 14,218 query images vs 12,612 gallery images.

| Metric | @5 | @10 | @15 |
|--------|----|-----|-----|
| **Recall** | 0.0082 | 0.0132 | 0.0155 |
| **NDCG** | 0.0057 | 0.0088 | 0.0104 |
| **mAP** | 0.0034 | 0.0040 | 0.0041 |

---

## Troubleshooting

**"Image not found" in results**
→ Check that **Images Directory** in the sidebar points to your local `cropped_images/` folder.

**Index not loading**
→ Verify all 3 required files exist in your index directory: `gallery_embs.npy`, `gallery_index.csv` (or `gallery_index_blip.csv`), and `config.json`.

**BLIP ITM slow on CPU**
→ Reduce **CLIP candidates to re-rank** to 20 in the sidebar. First run also downloads the model (~450 MB).

**BLIP ITM mode not available**
→ `transformers` must be installed: `pip install transformers`
