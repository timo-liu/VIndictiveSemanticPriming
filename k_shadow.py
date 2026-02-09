import pandas as pd
from scipy.cluster.vq import kmeans, vq, whiten
from sklearn.metrics import silhouette_score, adjusted_rand_score
import argparse
import numpy as np
from Definitions.Utils import *
import random
from matplotlib import pyplot as plt

def compute_kmeans(data,
				   model,
				   component,
				   embedding_dir,
				   graphs_dir,
				   prefix,
				   consolidate_reactions=True,
				   n=200,
				   seed=120):
	"""
	Load embeddings, compute prime–target differences, and run k-means
	"""

	# number of clusters
	ks = (
		pd.unique(data[['relation', 'isi']].values.ravel('K'))
		if not consolidate_reactions
		else data["relation"].unique()
	)
	k = len(ks)

	tokens = set(data["prime"]).union(set(data["target"]))
	emb_cache = load_or_build_embedding_cache(
		tokens,
		model=model,
		component=component,
		embedding_dir=embedding_dir
	)

	if consolidate_reactions:
		labels = data["relation"].to_numpy()
	else:
		labels = list(zip(data["relation"], data["isi"]))

	primes = data["prime"].apply(normalize_token).to_numpy()
	targets = data["target"].apply(normalize_token).to_numpy()
	pts = list(zip(primes, targets))
	combined = list(zip(pts, labels))

	rng = random.Random(seed)
	rng.shuffle(combined)
	pts_shuffled, labels_shuffled = zip(*combined)

	os.makedirs(graphs_dir, exist_ok=True)

	distortions = []

	silhouettes = []

	aris = []

	for i in range(len(pts_shuffled) // n):
		batch_pts = pts_shuffled[i * n:(i + 1) * n]

		observations = np.array([
			emb_cache[p] - emb_cache[t]
			for p, t in batch_pts
		])

		# normalize feature dimensions
		whitened = whiten(observations)

		codebook, distortion = kmeans(whitened, k)
		cluster_labels, actual_distortion = vq(whitened, codebook)
		if len(set(cluster_labels)) > 1:
			sil = silhouette_score(whitened, cluster_labels, metric="euclidean")
		else:
			sil = np.nan
		silhouettes.append(sil)

		batch_labels = labels_shuffled[i * n:(i + 1) * n]
		ari = adjusted_rand_score(batch_labels, cluster_labels)
		aris.append(ari)

		distortions.append(actual_distortion)

	# ---- plotting ----
	plt.figure(figsize=(8, 5))
	plt.plot(distortions, marker='o')
	plt.xlabel("Batch index")
	plt.ylabel("Distortion")
	plt.title(f"K-means distortion over batches ({prefix})")

	outpath = os.path.join(
		graphs_dir,
		f"{prefix}_kmeans_distortion.png"
	)
	plt.savefig(outpath, dpi=300, bbox_inches="tight")
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.plot(silhouettes, marker="o")
	plt.xlabel("Batch index")
	plt.ylabel("Silhouette score")
	plt.title(f"K-means silhouette over batches ({prefix})")
	plt.savefig(
		os.path.join(graphs_dir, f"{prefix}_kmeans_silhouette.png"),
		dpi=300,
		bbox_inches="tight"
	)
	plt.close()

	plt.figure(figsize=(8, 5))
	plt.plot(aris, marker="o")
	plt.xlabel("Batch index")
	plt.ylabel("Adjusted Rand Index")
	plt.title(f"K-means ARI over batches ({prefix})")
	plt.savefig(
		os.path.join(graphs_dir, f"{prefix}_kmeans_ari.png"),
		dpi=300,
		bbox_inches="tight"
	)
	plt.close()




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str, required=True)
	parser.add_argument('-i', '--hf_id', type=str, required=False, default="bert-base-uncased")
	parser.add_argument('-c', '--component', type=str, required=False, default="word_embeddings")
	parser.add_argument("--embeddingdir", default="D:/heat_embeddings")
	parser.add_argument("--graphs_dir", default="k_means_graphs")
	parser.add_argument("--no_consolidate", action="store_true")
	parser.add_argument("-n", type=int, default=200)
	parser.add_argument("--seed", type=int, default=120)
	args = parser.parse_args()

	# load data
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

	compute_kmeans(
		data=data,
		model=args.hf_id,
		component=args.component,
		embedding_dir=args.embeddingdir,
		graphs_dir=args.graphs_dir,
		prefix=prefix,
		consolidate_reactions=not args.no_consolidate,
		n=args.n,
		seed=args.seed
	)