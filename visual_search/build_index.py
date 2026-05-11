import torch
import open_clip
from PIL import Image
import pandas as pd
import numpy as np
import os
import argparse
from tqdm import tqdm
import json

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load CLIP model
    print(f"Loading CLIP model ViT-L-14...")
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
    model.to(device)
    model.eval()

    # Load metadata
    df = pd.read_csv(args.csv)
    img_dir = args.img_dir

    embeddings = []
    valid_indices = []

    print(f"Indexing {len(df)} images from {img_dir}...")
    
    # Try to find the image column
    img_col = None
    possible_cols = ['image_path', 'filepath', 'filename', 'image']
    for col in possible_cols:
        if col in df.columns:
            img_col = col
            break
    
    if img_col is None:
        # If not found, use the first column and hope for the best, or exit
        img_col = df.columns[0]
        print(f"Warning: Could not find obvious image path column. Using '{img_col}'")

    with torch.no_grad():
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            img_name = row[img_col]
            img_path = os.path.join(img_dir, img_name)
            
            if not os.path.exists(img_path):
                continue
            
            try:
                image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
                embedding = model.encode_image(image)
                embedding /= embedding.norm(dim=-1, keepdim=True)
                embeddings.append(embedding.cpu().numpy())
                valid_indices.append(idx)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    if not embeddings:
        print("No embeddings were created. Check your image directory and CSV.")
        return

    embeddings = np.vstack(embeddings)
    
    # Save results
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(os.path.join(args.out_dir, 'gallery_embs.npy'), embeddings)
    
    # Save a filtered CSV containing only indexed images
    indexed_df = df.iloc[valid_indices].copy()
    # Add absolute path for convenience in app, or relative? 
    # Let's keep it relative but ensure the app knows where the root is.
    indexed_df.to_csv(os.path.join(args.out_dir, 'gallery_index.csv'), index=False)
    
    config = {
        "model_name": "ViT-L-14",
        "pretrained": "openai",
        "num_indexed": len(valid_indices),
        "img_dir": os.path.abspath(img_dir),
        "img_col": img_col
    }
    with open(os.path.join(args.out_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    print(f"Done! Indexed {len(valid_indices)} images.")
    print(f"Saved index to {args.out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to cropped_metadata.csv")
    parser.add_argument("--img_dir", required=True, help="Path to cropped_images/ directory")
    parser.add_argument("--out_dir", default="./index", help="Output directory for index files")
    args = parser.parse_args()
    main(args)
