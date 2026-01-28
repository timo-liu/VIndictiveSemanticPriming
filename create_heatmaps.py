from Definitions.Utils import *
import argparse
from Definitions.GutsGorer import GutsGorer
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import numpy as np
import os
import pandas as pd
from tqdm.auto import tqdm


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------

import torch


def normalize_vector(vec, p=2, dim=0, eps=1e-12):
    """
    Normalize a vector (or tensor) to unit length along the specified dimension.

    Args:
        vec (torch.Tensor): Input vector or tensor.
        p (int): Norm degree (default 2 for L2 norm).
        dim (int): Dimension along which to normalize.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    norm = torch.norm(vec, p=p, dim=dim, keepdim=True).clamp(min=eps)
    return vec / norm


def plot_heatmaps(
    heat_maps,
    save_dir,
    *,
    figsize=(6, 5),
    cmap="viridis",
    prefix="nam",
    model="bert-base-uncased",
    component="word_embeddings"
):
    model_component = f"{model.replace('/', '_')}_{component}"
    os.makedirs(save_dir, exist_ok=True)

    for (isi, relation), matrix in tqdm(
        heat_maps.items(),
        desc="Plotting heatmaps",
        leave=False
    ):
        plt.figure(figsize=figsize)
        ax = sns.heatmap(matrix, cmap=cmap, square=True, cbar=True)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f"Heatmap: ISI={isi}, relation={relation}")
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                save_dir,
                f"{prefix}_heatmap_{model_component}_{isi}_{relation}.png"
            ),
            dpi=300
        )
        plt.close()


# ---------------------------------------------------------------------
# Embedding cache
# ---------------------------------------------------------------------

def normalize_token(tok):
    if not isinstance(tok, str) or not tok.strip():
        return None
    return tok.strip()  # optionally .lower()

def load_or_build_embedding_cache(tokens, *, model, component, embedding_dir):
    os.makedirs(embedding_dir, exist_ok=True)
    cache_path = os.path.join(
        embedding_dir,
        f"{model.replace('/', '_')}__{component}.npz"
    )

    # Load existing cache
    if os.path.exists(cache_path):
        cache = dict(np.load(cache_path, allow_pickle=True))
    else:
        cache = {}

    # Normalize and keep only valid tokens
    valid_tokens = [t for t in tokens if normalize_token(t) is not None]

    # Compute embeddings for missing tokens
    missing = [t for t in valid_tokens if t not in cache]
    if missing:
        gg = GutsGorer(model)
        print(f"Computing embeddings for {len(missing)} missing tokens...")

        for tok in tqdm(missing, desc="Computing embeddings"):
            try:
                if component == "word_embeddings":
                    vec = gg.compute_embedding(tok, component="word_embeddings")
                elif component.startswith("encoder_layer_"):
                    layer = int(component.split("_")[-1])
                    layers = get_or_compute_encoder_layers(tok, gg, {}, component)
                    vec = layers[layer]
                else:
                    raise ValueError(component)
            except Exception as e:
                print(f"Warning: could not embed token '{tok}': {e}")
                continue
            if isinstance(vec, np.ndarray):
                vec = torch.from_numpy(vec)
            cache[tok] = normalize_vector(vec)

        # Save updated cache immediately
        np.savez_compressed(cache_path, **cache)
        print(f"Cache updated: {cache_path}")

    return cache


# ---------------------------------------------------------------------
# Heatmap computation (with bootstrap)
# ---------------------------------------------------------------------

def difference_func(x, y):
    return abs(x - y) / 2


def heat_mapper(
    data,
    *,
    model,
    component,
    embedding_dir,
    n=None,
    bootstrap=1,
    bootstrap_samples=None
):
    heat_maps = {}
    relations = sorted(data["relation"].unique())
    isis = sorted(data["isi"].unique())

    tokens = set(data["prime"]).union(set(data["target"]))

    # Step 1: Load or build embedding cache (cached normalized vectors)
    emb_cache_np = load_or_build_embedding_cache(
        tokens,
        model=model,
        component=component,
        embedding_dir=embedding_dir
    )

    # Step 2: Convert numpy normalized embeddings to torch tensors for fast ops
    emb_cache = {tok: vec_np for tok, vec_np in emb_cache_np.items()}

    condition_iter = [
        (isi, rel)
        for isi in isis
        for rel in relations
    ]

    for isi, relation in tqdm(condition_iter, desc="Computing heatmaps"):
        subset = data[
            (data["isi"] == isi) &
            (data["relation"] == relation)
        ]

        if len(subset) < 2:
            continue

        n_eff = min(len(subset), n) if n else len(subset)
        bs_size = min(bootstrap_samples or n_eff, len(subset))

        boot_maps = []


        primes = subset["prime"].apply(normalize_token).to_list()
        targets = subset["target"].apply(normalize_token).to_list()

        valid_indices = []
        vectors = []

        # Build vectors representing prime-target pairs (normalized vectors)
        for idx, (p, t) in enumerate(zip(primes, targets)):
            if p in emb_cache and t in emb_cache:
                # Your vector representation for the pair — example: difference
                vec = emb_cache[p] - emb_cache[t]
                vec = torch.from_numpy(vec)
                # The vectors should remain normalized, normalize again just to be safe
                vec = torch.nn.functional.normalize(vec.unsqueeze(0), p=2, dim=1).squeeze(0)
                vectors.append(vec)
                valid_indices.append(idx)

        if len(vectors) < 2:
            continue

        emb_tensor = torch.stack(vectors)  # shape: (valid_pairs, emb_dim)

        for _ in tqdm(range(bootstrap), desc="Bootstraps", leave=False):
            sample_indices = np.random.choice(len(valid_indices), size=bs_size, replace=True)
            sample_emb = emb_tensor[sample_indices]

            # Cosine similarity matrix: since vectors are normalized, dot product = cosine similarity
            cos_sim_matrix = sample_emb @ sample_emb.T  # shape (bs_size, bs_size)

            diff_matrix = torch.abs(cos_sim_matrix.unsqueeze(0) - cos_sim_matrix.unsqueeze(1)) / 2
            diff_2d = diff_matrix.mean(dim=2).numpy()

            boot_maps.append(diff_2d)

        mean_map = np.mean(boot_maps, axis=0)

        heat_maps[(isi, relation)] = {
            "mean": mean_map,
            "bootstraps": boot_maps
        }

    return heat_maps



# ---------------------------------------------------------------------
# Load or compute
# ---------------------------------------------------------------------

def load_or_compute_heatmaps(
    data,
    *,
    model,
    component,
    embedding_dir,
    heatmap_dir,
    prefix,
    n,
    bootstrap,
    bootstrap_samples
):
    fname = os.path.join(
        heatmap_dir,
        f"{prefix}_{model.replace('/', '_')}__{component}.npz"
    )

    if os.path.exists(fname):
        with np.load(fname, allow_pickle=True) as f:
            heat_maps = {}
            for key in f.keys():
                isi, relation, kind = key.split("__", 2)
                heat_maps.setdefault((isi, relation), {})[kind] = f[key]
            return heat_maps

    heat_maps = heat_mapper(
        data,
        model=model,
        component=component,
        embedding_dir=embedding_dir,
        n=n,
        bootstrap=bootstrap,
        bootstrap_samples=bootstrap_samples
    )

    os.makedirs(heatmap_dir, exist_ok=True)

    save_dict = {}
    for (isi, relation), v in heat_maps.items():
        save_dict[f"{isi}__{relation}__mean"] = v["mean"]
        for i, hm in enumerate(v["bootstraps"]):
            save_dict[f"{isi}__{relation}__boot_{i:03d}"] = hm

    np.savez_compressed(fname, **save_dict)
    return heat_maps


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("-i", "--hf_id", default="bert-base-uncased")
    ap.add_argument("-c", "--component", default="word_embeddings")
    ap.add_argument("--heatmapdir", default="D:/heatmaps")
    ap.add_argument("--embeddingdir", default="D:/heat_embeddings")
    ap.add_argument("-n", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--bootstrap_samples", type=int, default=200)
    args = ap.parse_args()

    data = parse_data(
        args.data,
        extract_dict={
            "prime": "prime",
            "target": "target",
            "relation": "relation",
            "RT": "RT",
            "isi": "isi",
        }
    )

    prefix = "nam" if "nam" in args.data else "ldt"

    heat_maps = load_or_compute_heatmaps(
        data,
        model=args.hf_id,
        component=args.component,
        embedding_dir=args.embeddingdir,
        heatmap_dir=args.heatmapdir,
        prefix=prefix,
        n=args.n,
        bootstrap=args.bootstrap,
        bootstrap_samples=args.bootstrap_samples
    )

    plot_heatmaps(
        {k: v["mean"] for k, v in heat_maps.items()},
        save_dir=args.heatmapdir,
        prefix=prefix,
        model=args.hf_id,
        component=args.component
    )