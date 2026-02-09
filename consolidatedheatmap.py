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

# ---------------------------------------------------------------------
# RT-based dissimilarity (bootstrap-compatible)
# ---------------------------------------------------------------------

def rt_difference_matrix(rts):
	"""
	Compute pairwise normalized RT dissimilarity matrix.
	rts: 1D numpy array
	"""
	mx = np.nanmax(rts)
	mn = np.nanmin(rts)
	if mx == mn:
		return np.zeros((len(rts), len(rts)))

	norm = (rts - mn) / (mx - mn)
	diff = norm[:, None] - norm[None, :]
	return diff

def emb_difference_matrix(css):
	diff = (css[:, None] - css[None, :])/2
	return diff


# ---------------------------------------------------------------------
# Heatmap computation (with bootstrap)
# ---------------------------------------------------------------------

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
	rt_maps = {}

	relations = sorted(data["relation"].unique())
	isis = sorted(data["isi"].unique())

	tokens = set(data["prime"]).union(set(data["target"]))

	# --- Embedding cache ---
	emb_cache = load_or_build_embedding_cache(
		tokens,
		model=model,
		component=component,
		embedding_dir=embedding_dir
	)

	for isi in tqdm(isis, desc="Processing ISIs"):
		for relation in tqdm(relations, desc=f"Relations for ISI={isi}", leave=False):
			subset = data[
				(data["isi"] == isi) &
				(data["relation"] == relation)
			].reset_index(drop=True)

			if len(subset) < 2:
				continue

			subset = subset[: n if n else len(subset)]
			n_eff = len(subset)
			bs_size = min(bootstrap_samples or n_eff, n_eff)

			primes = subset["prime"].apply(normalize_token).to_numpy()
			targets = subset["target"].apply(normalize_token).to_numpy()
			rts = subset["RT"].to_numpy(dtype=float)

			vectors = []
			valid_rt = []

			for p, t, rt in zip(primes, targets, rts):
				if p in emb_cache and t in emb_cache and not np.isnan(rt):
					prim = emb_cache[p]
					targ = emb_cache[t]
					prim = prim if isinstance(prim, torch.Tensor) else torch.from_numpy(prim)
					targ = targ if isinstance(targ, torch.Tensor) else torch.from_numpy(targ)
					cos_sim = torch.dot(prim, targ).item()
					vectors.append(cos_sim)
					valid_rt.append(rt)

			if len(vectors) < 2:
				continue

			emb_tensor = np.asarray(vectors)
			valid_rt = np.asarray(valid_rt)

			emb_boot = []
			rt_boot = []

			for _ in tqdm(range(bootstrap), desc="Bootstraps", leave=False):
				idx = np.random.choice(len(valid_rt), size=bs_size, replace=True)

				# --- EMBEDDINGS ---
				sample_emb = emb_tensor[idx]
				emb_boot.append(emb_difference_matrix(sample_emb))

				# --- RTs (same indices!) ---
				rt_sample = valid_rt[idx]

				rt_boot.append(rt_difference_matrix(rt_sample))

			heat_maps[(isi, relation)] = {
				"mean": np.mean(emb_boot, axis=0),
				"bootstraps": emb_boot
			}

			rt_maps[(isi, relation)] = {
				"mean": np.mean(rt_boot, axis=0),
				"bootstraps": rt_boot
			}

	return heat_maps, rt_maps


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
	emb_path = os.path.join(
		heatmap_dir,
		f"{prefix}_{model.replace('/', '_')}__{component}_EMB.npz"
	)

	rt_path = os.path.join(
		heatmap_dir,
		f"{prefix}_{model.replace('/', '_')}__{component}_RT.npz"
	)

	if os.path.exists(emb_path) and os.path.exists(rt_path):
		def load_npz(path):
			with np.load(path, allow_pickle=True) as f:
				out = {}
				for k in f:
					isi, rel, kind = k.split("__", 2)
					out.setdefault((isi, rel), {})[kind] = f[k]
				return out

		return load_npz(emb_path), load_npz(rt_path)

	emb_maps, rt_maps = heat_mapper(
		data,
		model=model,
		component=component,
		embedding_dir=embedding_dir,
		n=n,
		bootstrap=bootstrap,
		bootstrap_samples=bootstrap_samples
	)

	os.makedirs(heatmap_dir, exist_ok=True)

	def save_npz(path, maps):
		save = {}
		for (isi, rel), v in maps.items():
			save[f"{isi}__{rel}__mean"] = v["mean"]
			for i, b in enumerate(v["bootstraps"]):
				save[f"{isi}__{rel}__boot_{i:03d}"] = b
		np.savez_compressed(path, **save)

	save_npz(emb_path, emb_maps)
	save_npz(rt_path, rt_maps)

	return emb_maps, rt_maps


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

	emb_maps, rt_maps = load_or_compute_heatmaps(
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
		{k: v["mean"] for k, v in emb_maps.items()},
		save_dir=args.heatmapdir,
		prefix=prefix,
		model=args.hf_id,
		component=args.component
	)
