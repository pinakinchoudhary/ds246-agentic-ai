import pandas as pd
import numpy as np
import torch
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
import faiss
import clip
from datasets import load_dataset

# Keep things clean by ignoring warnings for this demo
warnings.filterwarnings('ignore')

# Set up the environment
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Setting up shop on: {device}")

# Load the CLIP model
# We're using ViT-B/32 here because it's a good balance of speed and accuracy
model, preprocess = clip.load("ViT-B/32", device=device)

# ---------------------------------------------------------
# Data Loading & Prep
# ---------------------------------------------------------

# We'll pull the Amazon "All_Beauty" dataset from Hugging Face.
# It's a nice manageable size for testing without melting the RAM.
print("Grabbing the dataset...")
dataset = load_dataset("McAuley-Lab/Amazon-Reviews-2023", 
                       "raw_meta_All_Beauty",
                       split="full",
                       trust_remote_code=True)

df_products = dataset.to_pandas()

# Time to clean up the data. 
# We only care about products that actually have images and decent titles.
df_products = df_products[df_products['images'].notna()]
df_products = df_products[df_products['title'].notna()]

# Amazon stores images in a list of dictionaries, so let's extract the large URL.
df_products['image_url'] = df_products['images'].apply(
    lambda x: x[0]['large'] if isinstance(x, list) and len(x) > 0 and 'large' in x[0] else None
)

# drop rows where we couldn't find a URL or the title is too short to be useful
df_products = df_products[df_products['image_url'].notna()]
df_products = df_products[df_products['title'].str.len() > 10]

# For this script, we'll just slice a sample so it runs quickly.
SAMPLE_SIZE = 5000 
df_sample = df_products.sample(n=min(SAMPLE_SIZE, len(df_products)), random_state=42).reset_index(drop=True)
print(f"Data prepped. Working with {len(df_sample)} products.")

# ---------------------------------------------------------
# Image Encoding
# ---------------------------------------------------------

def download_image(url, timeout=5):
    """Simple helper to grab an image from a URL without crashing."""
    try:
        response = requests.get(url, timeout=timeout)
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception:
        return None

print("Starting the embedding process. This might take a minute...")

image_embeddings = []
valid_indices = []
batch_size = 32

# Loop through the dataset in chunks
for idx in tqdm(range(0, len(df_sample), batch_size)):
    batch_df = df_sample.iloc[idx:idx+batch_size]
    batch_images = []
    batch_valid_idx = []
    
    # Download images for the current batch
    for i, row in batch_df.iterrows():
        img = download_image(row['image_url'])
        if img:
            batch_images.append(img)
            batch_valid_idx.append(i)
    
    if not batch_images:
        continue
    
    # Run the batch through CLIP
    # We normalize the features right away so we can use dot product later
    image_inputs = torch.stack([preprocess(img) for img in batch_images]).to(device)
    
    with torch.no_grad():
        features = model.encode_image(image_inputs)
        features = features / features.norm(dim=-1, keepdim=True)
        image_embeddings.append(features.cpu())
        valid_indices.extend(batch_valid_idx)

# Stack everything into one big tensor
image_embeddings = torch.cat(image_embeddings, dim=0)
df_valid = df_sample.loc[valid_indices].reset_index(drop=True)

# ---------------------------------------------------------
# FAISS Indexing (The Fast Search Part)
# ---------------------------------------------------------

print("Building the FAISS index for high-speed searching...")
# Convert to float32 because FAISS is picky
embeddings_np = image_embeddings.numpy().astype('float32')

# Using Inner Product (IP) index, which is equivalent to Cosine Similarity 
# since we normalized the vectors earlier.
dimension = embeddings_np.shape[1]
index = faiss.IndexFlatIP(dimension)
index.add(embeddings_np)

# ---------------------------------------------------------
# Search Logic
# ---------------------------------------------------------

def search_products(query_text, faiss_index, products_df, top_k=5):
    """
    Standard Text-to-Image search.
    User types 'red lipstick', we find images that look like red lipstick.
    """
    text_tokens = clip.tokenize([query_text], truncate=True).to(device)
    
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    query_np = text_features.cpu().numpy().astype('float32')
    similarities, indices = faiss_index.search(query_np, top_k)
    
    results = []
    for idx, sim in zip(indices[0], similarities[0]):
        results.append({
            'title': products_df.iloc[idx]['title'],
            'image_url': products_df.iloc[idx]['image_url'],
            'score': float(sim)
        })
    
    return pd.DataFrame(results)

def visualize_search(query, results_df):
    """Quick helper to show the actual images we found."""
    fig, axes = plt.subplots(1, len(results_df), figsize=(15, 3))
    if len(results_df) == 1: axes = [axes]
    
    for idx, (i, row) in enumerate(results_df.iterrows()):
        img = download_image(row['image_url'])
        if img:
            axes[idx].imshow(img)
        axes[idx].set_title(f"{row['title'][:30]}...\nScore: {row['score']:.3f}", fontsize=8)
        axes[idx].axis('off')
    
    plt.suptitle(f"Search: '{query}'", fontsize=12, fontweight='bold')
    plt.show()

# ---------------------------------------------------------
# Running some Tests
# ---------------------------------------------------------

test_queries = [
    "red lipstick",
    "moisturizer for dry skin",
    "makeup brush set"
]

print("\n--- Running Demo Searches ---")

for query in test_queries:
    print(f"\nLooking for: {query}")
    results = search_products(query, index, df_valid, top_k=5)
    
    # Just printing the titles for the log
    for i, row in results.iterrows():
        print(f"  > {row['title'][:80]} (Score: {row['score']:.3f})")
    
    # Uncomment this if you want to see the actual images pop up
    # visualize_search(query, results)

print("\nScript finished successfully.")