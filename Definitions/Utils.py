from typing import Dict, Optional
import pandas as pd
import os
import numpy as np

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