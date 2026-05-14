# Visual Product Search

Streamlit demo for DeepFashion In-Shop retrieval with three modes:

| Condition | Runtime behavior |
| --- | --- |
| A | Frozen CLIP image-to-image search |
| B | Frozen CLIP + BLIP caption text fusion/reranking |
| C | Fine-tuned CLIP + BLIP caption text fusion/reranking |

The current indexes use `ViT-B-32` / `openai`, so all gallery embeddings are 512-dimensional.

## What's Included (Committed to Git)

```text
visual_search/
├── app.py                             # Main Streamlit application
├── build_index.py                     # Script to build embeddings
├── requirements.txt                   # Python dependencies
├── README.md                          # This file
├── data/
│   └── split_images/                  # ✓ INCLUDED | Gallery, query, and train images
├── index/
│   ├── config.json                    # ✓ INCLUDED | Model configuration
│   ├── gallery_embs.npy               # ✓ INCLUDED | Frozen CLIP embeddings
│   ├── gallery_embs_finetuned.npy     # ✓ INCLUDED | Fine-tuned CLIP embeddings
│   ├── gallery_text_embs.npy          # ✓ INCLUDED | BLIP caption embeddings
│   ├── gallery_index_blip.csv         # ✓ INCLUDED | Gallery metadata + captions
│   └── gallery_index.csv              # ✓ INCLUDED | Plain metadata
└── checkpoints/
    └── .gitkeep                       # Placeholder
```

**These files and directories are all tracked in the repository.**

## What's Excluded (Ignored by .gitignore)

```text
✗ Local environments:
  - visual_search/.conda/              # Local Conda environment
  - .venv/, venv/                      # Virtual environments

✗ Archive data:
  - visual_search/data/archive/        # Old/archived dataset

✗ Model weights (except clip_finetuned.pt):
  - *.pt, *.pth, *.ckpt               # PyTorch model checkpoints (YOLO, etc.)
  - *.onnx, *.bin, *.safetensors      # Other model formats

✗ Python cache:
  - __pycache__/                       # Python bytecode
  - *.pyc, *.pyo, *.pyd               # Compiled Python files

✗ OS files:
  - .DS_Store                          # macOS metadata
```

**Only `clip_finetuned.pt` needs to be added to `checkpoints/` after cloning.**

## Required Local Artifacts (Only After Cloning)

After cloning the repo, you only need to add one file:

```text
visual_search/checkpoints/
└── clip_finetuned.pt              # Only file to add | Required for Condition C | Download/train separately
```

**All other data files are already in the repository:**
- `split_images/` (gallery, query, train images)
- All `.npy` embedding files
- All `.csv` metadata files

**Note:** The checkpoint must be row-aligned with the embeddings in `gallery_embs_finetuned.npy`.

## Setup

Create and activate an environment. Python 3.11 is recommended.

```bash
cd visual_search
conda create -p ./.conda python=3.11 -y
conda activate ./.conda
python -m pip install -r requirements.txt
```

## Run

```bash
cd visual_search
conda activate ./.conda
python -m streamlit run app.py
```

Use these sidebar paths:

```text
Index Directory  = ./index
Images Directory = ./data/split_images/gallery
```

## Sidebar Modes

### Condition A

Uses:

```text
index/gallery_embs.npy
index/gallery_index_blip.csv or index/gallery_index.csv
data/split_images/gallery/
```

Search is direct CLIP image cosine similarity.

### Condition B

Uses:

```text
index/gallery_embs.npy
index/gallery_text_embs.npy
index/gallery_index_blip.csv
data/split_images/gallery/
```

Recommended demo setting:

```text
Condition = B - Frozen CLIP + BLIP captions
B/C retrieval mode = Alpha fusion
alpha image weight = 0.7
```

Other B modes:

```text
BLIP Text Similarity rerank
BLIP ITM neural rerank
```

BLIP ITM downloads `Salesforce/blip-itm-base-coco` on first use and is slower on CPU.

### Condition C

Uses:

```text
checkpoints/clip_finetuned.pt
index/gallery_embs_finetuned.npy
index/gallery_text_embs.npy
index/gallery_index_blip.csv
data/split_images/gallery/
```

The checkpoint and `gallery_embs_finetuned.npy` must come from the same fine-tuned CLIP run.

## Artifact Checklist

Before running, verify:

```text
gallery_embs.npy            shape = (12612, 512)
gallery_embs_finetuned.npy  shape = (12612, 512)
gallery_text_embs.npy       shape = (12612, 512)
gallery_index_blip.csv      rows  = 12612
data/split_images/gallery   images = 12612
```

## Troubleshooting

**Dimension mismatch during search**

Check `index/config.json`. The current indexes expect:

```json
{
  "model": "ViT-B-32",
  "pretrained": "openai"
}
```

**Condition C fails to load**

Make sure `visual_search/checkpoints/clip_finetuned.pt` is present and was saved from the same model used to build `gallery_embs_finetuned.npy`.

**Images not found**

Set the Streamlit sidebar to:

```text
Images Directory = ./data/split_images/gallery
```

The app resolves absolute Kaggle-style paths by falling back to the image basename in this local folder.
