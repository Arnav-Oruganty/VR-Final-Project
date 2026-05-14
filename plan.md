# 🔗 Integrating Conditions B & C — Visual Product Search

This guide tells your teammates exactly what to do to plug Conditions **B** and **C** into the existing codebase, which currently only runs **Condition A** (vision-only CLIP).

---

## Quick Summary of the Three Conditions

| ID | Config | What's different |
|----|--------|-----------------|
| **A** | Vision-only CLIP (α = 1) | Baseline — already implemented in `app.py` |
| **B** | Frozen CLIP + Frozen BLIP-2 | Re-rank CLIP results using BLIP-2 captions (no training required) |
| **C** | Fine-tuned CLIP + Frozen BLIP-2 | Same as B, but CLIP vision encoder has been contrastively fine-tuned |

> [!IMPORTANT]
> **B and C share the exact same app.py retrieval logic.** The only difference is **which gallery embeddings file you load**. For C you need to first fine-tune CLIP and rebuild the index with those fine-tuned weights.

---

## What's Already Done (Condition A)

The current `app.py`:
1. Loads frozen `ViT-L-14` (openai) via `open_clip`
2. YOLO-crops the query image
3. Encodes query with `clip_model.encode_image()`
4. Does cosine similarity against `gallery_embs.npy`
5. Returns top-K results

The current `build_index.py` builds `gallery_embs.npy` (pure CLIP vision embeddings, α=1).

---

## Part 1 — Integrating Condition B (Frozen CLIP + BLIP-2 Re-ranking)

Condition B already has **partial support** in the current `app.py` via the `gallery_text_embs.npy` path — but the sidebar dropdown and re-ranking logic are missing. Here's how to complete it.

### Step 1 — Produce `gallery_text_embs.npy` (on Kaggle/GPU)

This file encodes BLIP-2 captions using the CLIP **text** encoder.  
Run this on Kaggle (needs GPU for speed, but works on CPU too):

```python
# blip_build_text_index.py  ← run on Kaggle
import torch, open_clip, pandas as pd, numpy as np, os
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, _, _ = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
tokenizer   = open_clip.get_tokenizer('ViT-L-14')
model.to(device).eval()

# Load the captions CSV that was already generated with BLIP-2
df = pd.read_csv('captions (1).csv')   # columns: resolved_filename, clean_caption, item_id

text_embs = []
with torch.no_grad():
    for caption in tqdm(df['clean_caption'].fillna('')):
        tokens = tokenizer([caption]).to(device)
        emb    = model.encode_text(tokens)
        emb    = emb / emb.norm(dim=-1, keepdim=True)
        text_embs.append(emb.cpu().numpy())

text_embs = np.vstack(text_embs)
np.save('gallery_text_embs.npy', text_embs)
df.to_csv('gallery_index_blip.csv', index=False)
print("Done:", text_embs.shape)
```

Download `gallery_text_embs.npy` and `gallery_index_blip.csv` and place them in `visual_search/index/`.

### Step 2 — Update `app.py` to Add the Re-ranking Sidebar

Replace the current `app.py` search block with the version below. The key additions are:
- A **Re-ranking Mode** dropdown in the sidebar
- A **BLIP Text Similarity** mode (fast, no new model — just cosine sim on text embeddings)  
- A **BLIP ITM** mode (neural, uses `Salesforce/blip-itm-base-coco`)

```python
# ── Sidebar (add these lines after the existing top_k slider) ─────────────────
rerank_mode = st.sidebar.selectbox(
    "Re-ranking Mode",
    ["None (CLIP only)", "BLIP Text Similarity", "BLIP ITM (neural)"],
    index=0
)
blip_candidates = st.sidebar.slider("CLIP candidates to re-rank", 10, 200, 50)
```

```python
# ── Load gallery text embeddings (lazy, only when needed) ─────────────────────
@st.cache_resource
def load_text_embs(index_dir):
    path = os.path.join(index_dir, 'gallery_text_embs.npy')
    return np.load(path) if os.path.exists(path) else None

@st.cache_resource
def load_blip_itm():
    from transformers import BlipProcessor, BlipForImageTextRetrieval
    processor = BlipProcessor.from_pretrained("Salesforce/blip-itm-base-coco")
    model_itm = BlipForImageTextRetrieval.from_pretrained("Salesforce/blip-itm-base-coco")
    model_itm.eval()
    return processor, model_itm
```

```python
# ── Inside the "🔍 Search" button block, after CLIP retrieval ─────────────────

# Stage 1: CLIP visual retrieval (always runs)
n_candidates = blip_candidates if rerank_mode != "None (CLIP only)" else top_k
clip_sims     = np.dot(gallery_embs, q_img_emb.T).flatten()
cand_indices  = np.argsort(clip_sims)[::-1][:n_candidates].tolist()

# Stage 2: Re-ranking
if rerank_mode == "BLIP Text Similarity":
    text_embs = load_text_embs(INDEX_DIR)
    if text_embs is not None:
        cand_text_embs = text_embs[cand_indices]          # (n_candidates, 768)
        rerank_sims    = np.dot(cand_text_embs, q_img_emb.T).flatten()
        order          = np.argsort(rerank_sims)[::-1][:top_k]
        top_indices    = [cand_indices[i] for i in order]
        top_scores     = [float(rerank_sims[i]) for i in order]
    else:
        st.warning("gallery_text_embs.npy not found — falling back to CLIP only.")
        top_indices = cand_indices[:top_k]
        top_scores  = [float(clip_sims[i]) for i in top_indices]

elif rerank_mode == "BLIP ITM (neural)":
    with st.spinner("Loading BLIP-ITM model (first run ~450 MB download)…"):
        processor, itm_model = load_blip_itm()
    itm_scores = []
    for idx in cand_indices:
        caption = gallery_df.iloc[idx].get('clean_caption', '')
        inputs  = processor(images=query_crop, text=caption, return_tensors="pt")
        with torch.no_grad():
            score = itm_model(**inputs).itm_score
            itm_scores.append(float(score[0][1]))   # prob of "match"
    order       = np.argsort(itm_scores)[::-1][:top_k]
    top_indices = [cand_indices[i] for i in order]
    top_scores  = [itm_scores[i] for i in order]

else:  # CLIP only
    top_indices = cand_indices[:top_k]
    top_scores  = [float(clip_sims[i]) for i in top_indices]
```

> [!TIP]
> The **BLIP Text Similarity** mode is almost free — it's just a second cosine similarity using the already-computed caption embeddings. Use it as your primary B-condition demo.

---

## Part 2 — Integrating Condition C (Fine-tuned CLIP)

Condition C is identical to B in the **app**, except `gallery_embs.npy` is built from a **fine-tuned** CLIP checkpoint instead of the frozen one.

### Step 1 — Fine-tune CLIP on Kaggle (GPU required)

Create a new Kaggle notebook with this training script:

```python
# clip_finetune.py  ← run on Kaggle GPU
import torch, torch.nn.functional as F
import open_clip, pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os, random

SEED = 22116   # use your roll number as seed
random.seed(SEED); torch.manual_seed(SEED)

device = "cuda"
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
model.to(device)

# ── Freeze everything except the last 4 transformer blocks ────────────────────
for name, param in model.visual.named_parameters():
    param.requires_grad = False          # freeze all first
    if any(f"transformer.resblocks.{i}" in name for i in range(20, 24)):
        param.requires_grad = True       # unfreeze last 4 blocks
# Keep text encoder fully frozen
for param in model.transformer.parameters():
    param.requires_grad = False

# ── Dataset: sample pairs of same item_id (positives) ─────────────────────────
class InShopDataset(Dataset):
    def __init__(self, csv_path, img_dir, preprocess, split='train'):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        self.df = df
        self.img_dir = img_dir
        self.preprocess = preprocess
        # Group by item_id
        self.groups = df.groupby('item_id')['resolved_filename'].apply(list).to_dict()
        self.item_ids = [k for k, v in self.groups.items() if len(v) >= 2]

    def __len__(self): return len(self.item_ids)

    def __getitem__(self, i):
        iid  = self.item_ids[i]
        imgs = random.sample(self.groups[iid], 2)
        a    = self.preprocess(Image.open(os.path.join(self.img_dir, imgs[0])).convert('RGB'))
        b    = self.preprocess(Image.open(os.path.join(self.img_dir, imgs[1])).convert('RGB'))
        return a, b

# ── Contrastive loss (InfoNCE / NT-Xent) ─────────────────────────────────────
def contrastive_loss(a_emb, b_emb, temp=0.07):
    a = F.normalize(a_emb, dim=-1)
    b = F.normalize(b_emb, dim=-1)
    logits = torch.matmul(a, b.T) / temp
    labels = torch.arange(len(a), device=a.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

# ── Training loop ─────────────────────────────────────────────────────────────
TRAIN_CSV = '/kaggle/input/preprocessed-deepfashion/cropped_metadata.csv'
IMG_DIR   = '/kaggle/input/preprocessed-deepfashion/cropped_images'

dataset = InShopDataset(TRAIN_CSV, IMG_DIR, preprocess, split='train')
loader  = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)

for epoch in range(5):
    model.train()
    total_loss = 0
    for a, b in loader:
        a, b = a.to(device), b.to(device)
        with torch.cuda.amp.autocast():
            ea = model.encode_image(a)
            eb = model.encode_image(b)
            loss = contrastive_loss(ea, eb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1} — Loss: {total_loss/len(loader):.4f}")

torch.save(model.state_dict(), 'clip_finetuned.pt')
print("Saved clip_finetuned.pt")
```

> [!NOTE]
> Each team member should use their roll number as the `SEED`. The project asks for **mean ± std over 3–4 seeds**.

### Step 2 — Rebuild the Gallery Index with Fine-tuned Weights

After training, rebuild `gallery_embs.npy` using the saved checkpoint:

```python
# After training on Kaggle, rebuild index
model.load_state_dict(torch.load('clip_finetuned.pt'))
model.eval()
# ... then run build_index.py logic with this model
# Save output as gallery_embs_finetuned.npy
```

Or run `build_index.py` with a small modification to load your checkpoint:

```bash
# In build_index.py, add this after creating the model:
# model.load_state_dict(torch.load('clip_finetuned.pt', map_location=device))

python build_index.py \
  --csv data/archive/all_metadata.csv \
  --img_dir data/archive/cropped_images \
  --out_dir ./index_condition_c
```

Download `gallery_embs.npy` from `index_condition_c/` and place it as `index/gallery_embs_finetuned.npy`.

### Step 3 — Add Condition Selector to `app.py`

Add a dropdown to switch between A/B/C conditions:

```python
# In sidebar
condition = st.sidebar.selectbox(
    "Condition",
    ["A — Vision-only CLIP (frozen)", "B — Frozen CLIP + BLIP", "C — Fine-tuned CLIP + BLIP"]
)

# When loading embeddings:
if "C" in condition:
    emb_file = 'gallery_embs_finetuned.npy'
else:
    emb_file = 'gallery_embs.npy'
gallery_embs = np.load(os.path.join(INDEX_DIR, emb_file))
```

---

## File Checklist

| File | Who produces it | Where it goes |
|------|----------------|---------------|
| `gallery_embs.npy` | `a-vision-only-clip.ipynb` (Kaggle) | `index/` |
| `gallery_index.csv` | `a-vision-only-clip.ipynb` (Kaggle) | `index/` |
| `captions (1).csv` | BLIP-2 captioning notebook (Kaggle) | `visual_search/` |
| `gallery_text_embs.npy` | `blip_build_text_index.py` (Kaggle) | `index/` |
| `gallery_index_blip.csv` | `blip_build_text_index.py` (Kaggle) | `index/` |
| `clip_finetuned.pt` | `clip_finetune.py` (Kaggle GPU) | local only |
| `gallery_embs_finetuned.npy` | `build_index.py` with fine-tuned model | `index/` |

---

## α Value for Fused Embeddings (Optional Enhancement)

The project spec uses a fused embedding `v = α·ϕ_V(x) + (1-α)·ϕ_T(c)`. For conditions B and C you can experiment with **two α values** (e.g., α=0.7 and α=0.3):

```python
alpha = st.sidebar.slider("α (image weight)", 0.0, 1.0, 0.7, step=0.1)

# At search time:
q_img_emb_norm = F.normalize(torch.from_numpy(q_img_emb), dim=-1).numpy()
# Fuse with query text embedding (if you have it — optional)
# fused = alpha * q_img_emb_norm + (1-alpha) * q_text_emb_norm
# For app demo, α=1 = condition A, α<1 blends with gallery text side
```

> [!TIP]
> The simplest interpretation for the app: use **α** to weight between `gallery_embs.npy` (image side) and `gallery_text_embs.npy` (text side) when computing similarity. This gives you the full B-condition ablation for free.

---

## Summary: What Each Teammate Needs to Do

| Task | Person | Tool |
|------|--------|------|
| Generate BLIP-2 captions | One person | Kaggle GPU notebook |
| Build `gallery_text_embs.npy` | One person | `blip_build_text_index.py` on Kaggle |
| Fine-tune CLIP (multiple seeds) | All members (one seed each) | `clip_finetune.py` on Kaggle |
| Rebuild index with fine-tuned CLIP | All members | `build_index.py` modified |
| Update `app.py` with re-ranking UI | One person | local |
| Run metrics (Recall/NDCG/mAP) | One person | existing eval notebook |
