import argparse
import json
import os
import pickle

import hnswlib
import numpy as np
import pandas as pd


ALPHA_IMAGE = 0.7
ALPHA_TEXT = 0.3


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def load_gallery_metadata(index_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(index_dir, "gallery_index_blip.csv")
    if not os.path.exists(csv_path):
        csv_path = os.path.join(index_dir, "gallery_index.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError("Missing gallery_index_blip.csv / gallery_index.csv")
    return pd.read_csv(csv_path)


def add_derived_metadata(gallery_df: pd.DataFrame) -> pd.DataFrame:
    gallery_df = gallery_df.copy()
    if "image_path" in gallery_df.columns and "original_file_name" not in gallery_df.columns:
        gallery_df["original_file_name"] = gallery_df["image_path"].map(
            lambda value: os.path.basename(str(value))
        )
    return gallery_df


def build_hnsw(fused: np.ndarray, out_path: str, m: int, ef_construction: int, ef_search: int):
    dim = fused.shape[1]
    num = fused.shape[0]

    index = hnswlib.Index(space="cosine", dim=dim)
    index.init_index(max_elements=num, ef_construction=ef_construction, M=m)
    index.add_items(fused, np.arange(num))
    index.set_ef(ef_search)
    index.save_index(out_path)


def main(args):
    img_emb_path = os.path.join(args.index_dir, "gallery_embs_finetuned.npy")
    text_emb_path = os.path.join(args.index_dir, "gallery_text_embs.npy")

    if not os.path.exists(img_emb_path):
        raise FileNotFoundError(f"Missing {img_emb_path}")
    if not os.path.exists(text_emb_path):
        raise FileNotFoundError(f"Missing {text_emb_path}")

    image_embs = np.load(img_emb_path).astype("float32")
    text_embs = np.load(text_emb_path).astype("float32")

    if image_embs.shape != text_embs.shape:
        raise ValueError(f"Shape mismatch: image={image_embs.shape}, text={text_embs.shape}")

    image_embs = l2_normalize(image_embs)
    text_embs = l2_normalize(text_embs)

    fused = (ALPHA_IMAGE * image_embs) + (ALPHA_TEXT * text_embs)
    fused = l2_normalize(fused).astype("float32")

    os.makedirs(args.index_dir, exist_ok=True)
    fused_path = os.path.join(args.index_dir, "gallery_fused_embs.npy")
    np.save(fused_path, fused)

    hnsw_path = os.path.join(args.index_dir, "gallery_fused_hnsw.bin")
    build_hnsw(fused, hnsw_path, args.m, args.ef_construction, args.ef_search)

    gallery_df = add_derived_metadata(load_gallery_metadata(args.index_dir))
    if len(gallery_df) != fused.shape[0]:
        raise ValueError(
            f"Row mismatch: gallery csv rows={len(gallery_df)}, fused embeddings={fused.shape[0]}"
        )

    metadata_pickle_path = os.path.join(args.index_dir, "gallery_metadata.pkl")
    with open(metadata_pickle_path, "wb") as f:
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

    config_path = os.path.join(args.index_dir, "config.json")
    config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    config.update(
        {
            "alpha_image": ALPHA_IMAGE,
            "alpha_text": ALPHA_TEXT,
            "embedding_dim": int(fused.shape[1]),
            "fused_dim": int(fused.shape[1]),
            "num_indexed": int(fused.shape[0]),
            "index_type": "hnswlib_cosine",
            "hnsw_index_file": "gallery_fused_hnsw.bin",
            "fused_embeddings_file": "gallery_fused_embs.npy",
            "metadata_file": "gallery_metadata.pkl",
            "hnsw_m": args.m,
            "hnsw_ef_construction": args.ef_construction,
            "hnsw_ef_search": args.ef_search,
        }
    )

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print("Done.")
    print(f"Saved fused embeddings: {fused_path}")
    print(f"Saved HNSW index:      {hnsw_path}")
    print(f"Saved metadata pickle: {metadata_pickle_path}")
    print(f"Indexed items:         {fused.shape[0]}")
    print(f"Embedding dimension:   {fused.shape[1]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="One-time prereq: fuse 0.7*gallery_embs_finetuned + 0.3*gallery_text_embs, normalize, and build HNSW index"
    )
    parser.add_argument("--index_dir", default="./index", help="Directory containing gallery embeddings/csv")
    parser.add_argument("--m", type=int, default=32, help="HNSW M")
    parser.add_argument("--ef_construction", type=int, default=200, help="HNSW ef_construction")
    parser.add_argument("--ef_search", type=int, default=128, help="HNSW ef_search")
    main(parser.parse_args())
