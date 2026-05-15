# Visual Product Search Engine

**Visual Recognition Course – Final Project**

This repository contains the code and resources for a query-by-image product search system. The system retrieves visually and semantically similar products from a fashion catalog given a user-uploaded query image, addressing the common e-commerce issue of language mismatch and inconsistent metadata.

## Overview

The system operates on the **DeepFashion In-Shop Clothes Retrieval** dataset and consists of two primary pipelines:

1. **Offline Indexing Pipeline**: 
   - YOLO localization to crop primary clothing items.
   - BLIP-2 semantic captioning to generate descriptive text.
   - CLIP cross-modal fusion to combine visual and textual embeddings.
   - HNSW indexing for fast approximate nearest-neighbor search.

2. **Online Query Pipeline**:
   - Query image undergoes YOLO cropping and CLIP visual encoding.
   - HNSW candidate retrieval by cosine similarity (Top-15).
   - Semantic re-ranking using BLIP-2 Image-Text Matching (ITM) scores.

## Repository Structure

The project is structured around the three ablation conditions required for evaluation:

- `a-vision-only-clip.ipynb`
  **Condition A**: Pure vision-only baseline using pre-trained CLIP ($\alpha=1$). No semantic alignment or fine-tuning.

- `Frozen-CLIP-Frozen-BLIP/`
  **Condition B**: Frozen CLIP visual encoder + Frozen BLIP-2 semantic text fusion. Explores multimodal fusion without fine-tuning at $\alpha=0.5$ and $\alpha=0.7$.

- `Fine-Tuned-CLIP-Frozen-BLIP/`
  **Condition C**: Fine-tuned CLIP visual encoder + Frozen BLIP-2 semantic text fusion. Contains the `c_finetuning.py` script featuring 3-seed contrastive fine-tuning using InfoNCE loss, complete with HNSW indexing and BLIP-2 ITM re-ranking.

- `visual_search/`
  Contains the interactive Streamlit demonstration application (`app.py`). It accepts a user query image, performs YOLO detection, fetches matches from the pre-computed HNSW index, and displays similarity scores.

- **Preprocessing & Data Generation**:
  - `preproc-yolo.ipynb` & `preproc-gt-bbox.ipynb`: Bounding box generation and image cropping.
  - `blip-captioning-final.ipynb`: Generates semantic captions for the gallery using BLIP-2.

- **Documentation**:
  - `report.tex` / `VR-Final-Project.pdf`: The complete technical report outlining motivation, architecture, ablation results, and analysis.

## Evaluation Protocol

Results are evaluated across three core metrics for $K \in \{5, 10, 15\}$:
- **Recall@K**: Fraction of queries where a relevant item is retrieved.
- **NDCG@K**: Position-aware ranking quality.
- **mAP@K**: Mean Average Precision up to rank K.

All fine-tuning experiments (Condition C) report the mean $\pm$ standard deviation across three random seeds to ensure robustness.
