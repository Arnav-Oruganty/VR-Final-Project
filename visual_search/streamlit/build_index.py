import argparse
import json
import os
import pickle
from pathlib import Path

import hnswlib
import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_GALLERY_DIR = PROJECT_DIR / "data" / "split_images" / "gallery"
DEFAULT_CAPTIONS_CSV = PROJECT_DIR / "captions" / "gallery_index_blip.csv"
DEFAULT_CHECKPOINT = PROJECT_DIR / "clip-model" / "clip_finetuned.pt"
DEFAULT_INDEX_DIR = PROJECT_DIR / "index"

CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "openai"
ALPHA_IMAGE = 0.8
ALPHA_TEXT = 0.2


class ImagePathDataset(Dataset):
    def __init__(self, image_paths, preprocess):
        self.image_paths = list(image_paths)
        self.preprocess = preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        with Image.open(self.image_paths[idx]) as image:
            return self.preprocess(image.convert("RGB"))


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_finetuned_clip_model(checkpoint_path: Path, device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME,
        pretrained=CLIP_PRETRAINED,
        device=device,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state = checkpoint
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


def add_derived_metadata(gallery_df: pd.DataFrame) -> pd.DataFrame:
    gallery_df = gallery_df.copy()
    if "image_path" not in gallery_df.columns:
        raise ValueError("Caption CSV must include an image_path column.")
    if "caption" not in gallery_df.columns:
        raise ValueError("Caption CSV must include a caption column.")
    gallery_df["original_file_name"] = gallery_df["image_path"].map(
        lambda value: os.path.basename(str(value))
    )
    return gallery_df


def resolve_gallery_path(row, gallery_dir: Path) -> Path:
    image_path = Path(str(row["image_path"]))
    candidate = gallery_dir / str(row["original_file_name"])
    if candidate.exists():
        return candidate
    if image_path.is_absolute() and image_path.exists():
        return image_path
    return candidate


def load_gallery_metadata(captions_csv: Path, gallery_dir: Path) -> pd.DataFrame:
    gallery_df = add_derived_metadata(pd.read_csv(captions_csv))
    gallery_df["resolved_image_path"] = gallery_df.apply(
        lambda row: str(resolve_gallery_path(row, gallery_dir)),
        axis=1,
    )
    exists_mask = gallery_df["resolved_image_path"].map(lambda p: Path(p).exists())
    if not exists_mask.all():
        missing = gallery_df.loc[~exists_mask, "resolved_image_path"].head(10).tolist()
        raise FileNotFoundError(
            f"{len(gallery_df) - int(exists_mask.sum())} gallery images are missing. "
            f"First missing paths: {missing}"
        )
    return gallery_df.reset_index(drop=True)


def encode_images(
    model,
    preprocess,
    image_paths,
    batch_size: int,
    device: str,
    num_workers: int,
    use_amp: bool,
) -> np.ndarray:
    dataset = ImagePathDataset(image_paths, preprocess)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
    )
    batches = []
    for image_tensor in tqdm(loader, desc="Encoding gallery images"):
        image_tensor = image_tensor.to(device, non_blocking=device == "cuda")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=use_amp and device == "cuda",
        ):
            emb = model.encode_image(image_tensor)
        batches.append(F.normalize(emb.float(), dim=-1).cpu().numpy().astype("float32"))
    return np.concatenate(batches, axis=0)


def encode_captions(model, captions, batch_size: int, device: str) -> np.ndarray:
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    batches = []
    clean_captions = [
        caption if isinstance(caption, str) and caption.strip() else "a clothing item"
        for caption in captions
    ]
    for start in tqdm(range(0, len(clean_captions), batch_size), desc="Encoding captions"):
        batch = clean_captions[start : start + batch_size]
        tokens = tokenizer(batch).to(device)
        with torch.inference_mode():
            emb = model.encode_text(tokens)
        batches.append(F.normalize(emb.float(), dim=-1).cpu().numpy().astype("float32"))
    return np.concatenate(batches, axis=0)


def build_hnsw(fused: np.ndarray, out_path: Path, m: int, ef_construction: int, ef_search: int):
    dim = fused.shape[1]
    num = fused.shape[0]
    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=num, ef_construction=ef_construction, M=m)
    index.add_items(fused, np.arange(num))
    index.set_ef(ef_search)
    index.save_index(str(out_path))


def write_metadata_pickle(gallery_df: pd.DataFrame, path: Path, fused: np.ndarray):
    with path.open("wb") as f:
        pickle.dump(
            {
                "metadata": gallery_df.to_dict(orient="records"),
                "columns": gallery_df.columns.tolist(),
                "num_indexed": int(fused.shape[0]),
                "embedding_dim": int(fused.shape[1]),
                "row_alignment": "metadata row i matches embedding/index label i",
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def main(args):
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    use_amp = device == "cuda" and not args.no_amp
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU. If you expected GPU, install a CUDA-enabled PyTorch build.")
    gallery_dir = Path(args.gallery_dir)
    captions_csv = Path(args.captions_csv)
    checkpoint_path = Path(args.checkpoint_path)
    index_dir = Path(args.index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    gallery_df = load_gallery_metadata(captions_csv, gallery_dir)
    model, preprocess = load_finetuned_clip_model(checkpoint_path, device)

    image_embs = encode_images(
        model,
        preprocess,
        gallery_df["resolved_image_path"].tolist(),
        args.image_batch_size,
        device,
        args.num_workers,
        use_amp,
    )
    text_embs = encode_captions(
        model,
        gallery_df["caption"].tolist(),
        args.text_batch_size,
        device,
    )

    if image_embs.shape != text_embs.shape:
        raise ValueError(f"Shape mismatch: image={image_embs.shape}, text={text_embs.shape}")

    image_embs = l2_normalize(image_embs).astype("float32")
    text_embs = l2_normalize(text_embs).astype("float32")
    fused = l2_normalize((ALPHA_IMAGE * image_embs) + (ALPHA_TEXT * text_embs)).astype("float32")

    image_emb_path = index_dir / "gallery_embs_finetuned.npy"
    text_emb_path = index_dir / "gallery_text_embs.npy"
    fused_path = index_dir / "gallery_fused_embs.npy"
    hnsw_path = index_dir / "gallery_fused_hnsw.bin"
    metadata_pickle_path = index_dir / "gallery_metadata.pkl"
    config_path = index_dir / "config.json"

    np.save(image_emb_path, image_embs)
    np.save(text_emb_path, text_embs)
    np.save(fused_path, fused)
    build_hnsw(fused, hnsw_path, args.m, args.ef_construction, args.ef_search)
    write_metadata_pickle(gallery_df, metadata_pickle_path, fused)

    config = {
        "gallery_dir": str(gallery_dir),
        "captions_csv": str(captions_csv),
        "checkpoint_path": str(checkpoint_path),
        "model": CLIP_MODEL_NAME,
        "pretrained": CLIP_PRETRAINED,
        "alpha_image": ALPHA_IMAGE,
        "alpha_text": ALPHA_TEXT,
        "embedding_dim": int(fused.shape[1]),
        "fused_dim": int(fused.shape[1]),
        "num_indexed": int(fused.shape[0]),
        "top_k_values": [5, 10, 15],
        "index_type": "hnswlib_cosine",
        "hnsw_index_file": hnsw_path.name,
        "image_embeddings_file": image_emb_path.name,
        "text_embeddings_file": text_emb_path.name,
        "fused_embeddings_file": fused_path.name,
        "metadata_file": metadata_pickle_path.name,
        "metadata_source_file": captions_csv.name,
        "itm_model": "Salesforce/blip2-itm-vit-g",
        "hnsw_m": args.m,
        "hnsw_ef_construction": args.ef_construction,
        "hnsw_ef_search": args.ef_search,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("Done.")
    print(f"Gallery images:        {gallery_dir}")
    print(f"Caption metadata:      {captions_csv}")
    print(f"Checkpoint:            {checkpoint_path}")
    print(f"Saved image embeddings:{image_emb_path}")
    print(f"Saved text embeddings: {text_emb_path}")
    print(f"Saved fused embeddings:{fused_path}")
    print(f"Saved HNSW index:      {hnsw_path}")
    print(f"Saved metadata pickle: {metadata_pickle_path}")
    print(f"Indexed items:         {fused.shape[0]}")
    print(f"Embedding dimension:   {fused.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Build the gallery index by encoding gallery images and captions with "
            "fine-tuned CLIP ViT-L-14, fusing 0.8 image + 0.2 caption embeddings, "
            "and storing an HNSW cosine index plus metadata."
        )
    )
    parser.add_argument("--gallery_dir", default=str(DEFAULT_GALLERY_DIR))
    parser.add_argument("--captions_csv", default=str(DEFAULT_CAPTIONS_CSV))
    parser.add_argument("--checkpoint_path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--index_dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--image_batch_size", type=int, default=32)
    parser.add_argument("--text_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--ef_construction", type=int, default=200)
    parser.add_argument("--ef_search", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    main(parser.parse_args())
