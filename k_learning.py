from platform import architecture
import torch
import pandas as pd
from Definitions.k_model import k_model
import argparse
import numpy as np
from Definitions.Utils import *
import random
from matplotlib import pyplot as plt

def learn_k(data,
			model,
			component,
			embedding_dir,
			graphs_dir,
			prefix,
			consolidate_reactions=True,
			n=200,
			seed=120,
			epochs : int = 10):
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

	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.use_deterministic_algorithms(True)

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
		batch_labels = labels_shuffled[i * n:(i + 1) * n]
		batch_labels = to_categorical(batch_labels)


		architecture = np.linspace(observations[0].shape[0], len(ks), num=4 ,dtype=int)
		model = k_model(architecture)
		optimizer = nn.Adam(model.parameters(), lr=0.001)
		criterion = nn.CrossEntropyLoss()
		xy = list(zip(observations, batch_labels))
		train_split = int(0.8 * len(xy))
		val_test_split = train_split + (len(xy) - train_split)//2
		train_obs = xy[0: train_split]
		val_obs = xy[train_split: val_test_split]
		test_obs = xy[val_test_split:]
		train_losses = []
		val_losses = []

		for e in tqdm(range(epochs), desc=f"Epochs (batch {i})", leave=False):
			epoch_loss = 0.0
			epoch_val_loss = 0.0

			# ---- validation ----
			model.eval()
			with torch.no_grad():
				for x, y in val_obs:
					x = torch.tensor(x, dtype=torch.float32)
					y = torch.tensor(y, dtype=torch.long)
					output = model(x)
					loss = criterion(output.unsqueeze(0), y.unsqueeze(0))
					epoch_val_loss += loss.item()

			# ---- training ----
			model.train()
			for x, y in train_obs:
				x = torch.tensor(x, dtype=torch.float32)
				y = torch.tensor(y, dtype=torch.long)

				output = model(x)
				loss = criterion(output.unsqueeze(0), y.unsqueeze(0))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				epoch_loss += loss.item()

			epoch_loss /= len(train_obs)
			epoch_val_loss /= max(1, len(val_obs))

			print(
				f"[Batch {i:03d} | Epoch {e + 1:02d}] "
				f"train_loss={epoch_loss:.4f} "
				f"val_loss={epoch_val_loss:.4f}"
			)
			train_losses.append(epoch_loss)
			val_losses.append(epoch_val_loss)

	# ---- plotting: training vs validation loss ----
	plt.figure(figsize=(8, 5))

	epochs_range = range(1, len(train_losses) + 1)

	plt.plot(epochs_range, train_losses, marker="o", label="Train loss")
	plt.plot(epochs_range, val_losses, marker="o", label="Validation loss")

	plt.xlabel("Epoch")
	plt.ylabel("Cross-Entropy Loss")
	plt.title(f"Training vs Validation Loss ({prefix}, batch {i})")
	plt.legend()
	plt.grid(True)

	outpath = os.path.join(
		graphs_dir,
		f"{prefix}_batch_{i:03d}_loss_curve.png"
	)

	plt.savefig(outpath, dpi=300, bbox_inches="tight")
	plt.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str, required=True)
	parser.add_argument('-i', '--hf_id', type=str, required=False, default="bert-base-uncased")
	parser.add_argument('-c', '--component', type=str, required=False, default="word_embeddings")
	parser.add_argument("--embeddingdir", default="D:/heat_embeddings")
	parser.add_argument("--graphs_dir", default="learning_graphs")
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