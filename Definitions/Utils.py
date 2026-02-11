from typing import Dict, Optional
import pandas as pd
import os
import torch
import numpy as np
from Definitions.GutsGorer import GutsGorer
from tqdm import tqdm
from typing import List

def parse_data(
    path: str,
    extract_dict: Optional[Dict[str, str]] = None,
    sheet_name: Optional[str] = None
):

    if extract_dict is None:
        extract_dict = {
            "prime": "prime",
            "target": "target",
            "primecondition": "primecond",
            "RT": "target.RT",
            "accuracy": "target.ACC",
            "isi": "isi"
        }

    # ---------- Load data ----------
    if path.endswith((".xlsx", ".xls")):
        if sheet_name is None:
            df = pd.read_excel(path, sheet_name=0)
        else:
            df = pd.read_excel(path, sheet_name=sheet_name)

    elif path.endswith(".csv"):
        df = pd.read_csv(path)

    else:
        raise ValueError(f"Unsupported file type: {path}")

    # ---------- Extract columns ----------
    rdf = pd.DataFrame()
    for out_col, src_col in extract_dict.items():
        if src_col in df.columns:
            rdf[out_col] = df[src_col]
        else:
            print(f"Warning: '{src_col}' not found in {path}")

    return rdf


def load_embedding_cache(cache_path):
	if os.path.exists(cache_path):
		return dict(np.load(cache_path, allow_pickle=True))
	return {}

def save_embedding_cache(cache, cache_path):
	os.makedirs(os.path.dirname(cache_path), exist_ok=True)
	np.savez_compressed(cache_path, **cache)

def get_or_compute_encoder_layers(word, gg, cache, component, *, normalize=False):
	"""
	Return encoder-layer embeddings for a word.
	Shape: (n_layers, hidden_dim)

	NOTE:
	- Does NOT handle word_embeddings
	"""
	if word in cache:
		return cache[word]

	layer_embs = gg.compute_embedding(
		word,
		component=component,
		return_all=True
	)

	# stack → (layers, dim)
	embs = np.stack(layer_embs, axis=0)

	if normalize:
		norms = np.linalg.norm(embs, axis=1, keepdims=True)
		embs = embs / np.clip(norms, 1e-9, None)

	cache[word] = embs
	return embs

def is_blank(x):
	return pd.isna(x) or str(x).strip() == ""

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

def normalize_token(tok):
	if not isinstance(tok, str) or not tok.strip():
		return None
	return tok.strip()  # optionally .lower()

def to_categorical(list : List):
	uniques = set(list)
	unique_dict = {u : i for i, u in enumerate(uniques)}
	return [unique_dict[u] for u in list]