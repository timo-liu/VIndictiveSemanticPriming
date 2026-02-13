import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.vq import whiten
from sklearn.metrics import silhouette_score, adjusted_rand_score
import argparse
import numpy as np
from Definitions.Utils import *
import random
from matplotlib import pyplot as plt

def compute_agglomerative(data,
                          model,
                          component,
                          embedding_dir,
                          graphs_dir,
                          prefix,
                          consolidate_reactions=True,
                          n=200,
                          seed=120):
    """
    Load embeddings, compute prime–target differences,
    and run Agglomerative Clustering batch-wise.
    """

    # ---- number of clusters ----
    ks = (
        pd.unique(data[['relation', 'isi']].values.ravel('K'))
        if not consolidate_reactions
        else data["relation"].unique()
    )
    k = len(ks)

    # ---- embedding cache ----
    tokens = set(data["prime"]).union(set(data["target"]))
    emb_cache = load_or_build_embedding_cache(
        tokens,
        model=model,
        component=component,
        embedding_dir=embedding_dir
    )

    # ---- labels ----
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

    # ---- batch loop ----
    for i in range(len(pts_shuffled) // n):

        batch_pts = pts_shuffled[i * n:(i + 1) * n]

        observations = np.array([
            emb_cache[p] - emb_cache[t]
            for p, t in batch_pts
        ])

        whitened = whiten(observations)

        # ---- Agglomerative Clustering ----
        clustering = AgglomerativeClustering(
            n_clusters=k,
            metric="euclidean",
            linkage="ward"
        )

        cluster_labels = clustering.fit_predict(whitened)

        # ---- Silhouette ----
        if len(set(cluster_labels)) > 1:
            sil = silhouette_score(whitened, cluster_labels, metric="euclidean")
        else:
            sil = np.nan
        silhouettes.append(sil)

        # ---- ARI ----
        batch_labels = labels_shuffled[i * n:(i + 1) * n]
        ari = adjusted_rand_score(batch_labels, cluster_labels)
        aris.append(ari)

        # ---- Distortion (within-cluster SSE) ----
        dist = 0.0
        for c in np.unique(cluster_labels):
            cluster_points = whitened[cluster_labels == c]
            centroid = cluster_points.mean(axis=0)
            dist += np.sum((cluster_points - centroid) ** 2)

        distortions.append(dist)

    # ============================
    #            PLOTS
    # ============================

    # ---- Distortion ----
    plt.figure(figsize=(8, 5))
    plt.plot(distortions, marker='o')
    plt.xlabel("Batch index")
    plt.ylabel("Within-cluster SSE")
    plt.title(f"Agglomerative distortion over batches ({prefix})")
    plt.savefig(
        os.path.join(graphs_dir, f"{prefix}_aclust_distortion.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ---- Silhouette ----
    plt.figure(figsize=(8, 5))
    plt.plot(silhouettes, marker="o")
    plt.xlabel("Batch index")
    plt.ylabel("Silhouette score")
    plt.title(f"Agglomerative silhouette over batches ({prefix})")
    plt.savefig(
        os.path.join(graphs_dir, f"{prefix}_aclust_silhouette.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    # ---- ARI (Bar Chart) ----
    plt.figure(figsize=(8, 5))

    x_positions = np.arange(len(aris))
    plt.bar(x_positions, aris)

    plt.xlabel("Batch index")
    plt.ylabel("Adjusted Rand Index")
    plt.title(f"Agglomerative ARI over batches ({prefix})")

    plt.xticks(x_positions)  # show all batch indices

    plt.savefig(
        os.path.join(graphs_dir, f"{prefix}_aclust_ari.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str)
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

	compute_agglomerative(
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