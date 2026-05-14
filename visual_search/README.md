# Visual Product Search

Streamlit demo for DeepFashion In-Shop retrieval with three modes:

| Condition | Runtime behavior |
| --- | --- |
| A | Frozen CLIP image-to-image search |
| B | Frozen CLIP + BLIP caption text fusion/reranking |
| C | Fine-tuned CLIP + BLIP caption text fusion/reranking |

The current indexes use `ViT-B-32` / `openai`, so all gallery embeddings are 512-dimensional.

## Repository Files

Commit these files:

```text
visual_search/app.py
visual_search/build_index.py
visual_search/requirements.txt
visual_search/README.md
visual_search/index/config.json
visual_search/checkpoints/.gitkeep
```

Do not commit the dataset, `.npy` index files, model checkpoints, local environments, or YOLO weights. They are intentionally ignored by `.gitignore`.

## Required Local Artifacts

After cloning the repo, download or copy the artifacts into these exact paths:

```text
visual_search/
├── checkpoints/
│   └── clip_finetuned.pt              # Required for Condition C
├── data/
│   └── split_images/
│       ├── gallery/                   # Required at runtime
│       └── train/                     # Only needed for training/rebuilding C
└── index/
    ├── config.json                    # Committed, model config
    ├── gallery_embs.npy               # Frozen CLIP gallery image embeddings
    ├── gallery_embs_finetuned.npy     # Fine-tuned CLIP gallery image embeddings
    ├── gallery_text_embs.npy          # CLIP text embeddings of BLIP captions
    ├── gallery_index.csv              # Optional plain metadata
    └── gallery_index_blip.csv         # Gallery metadata + BLIP captions
```

The four gallery files must be row-aligned:

```text
gallery_index_blip.csv
gallery_embs.npy
gallery_embs_finetuned.npy
gallery_text_embs.npy
```

Row `i` in every file must describe the same gallery image.

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
