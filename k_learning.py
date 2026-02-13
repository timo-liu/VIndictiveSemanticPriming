from platform import architecture
import torch
import torch.nn as nn
import pandas as pd
from Definitions.k_model import k_model
import argparse
import numpy as np
from Definitions.Utils import *
import random
from tqdm import tqdm
from matplotlib import pyplot as plt

from platform import architecture
import torch
import torch.nn as nn
import pandas as pd
from Definitions.k_model import k_model
import argparse
import numpy as np
from Definitions.Utils import *
import random
from matplotlib import pyplot as plt
from tqdm import tqdm
import os


def learn_k(data,
			model,
			component,
			embedding_dir,
			graphs_dir,
			prefix,
			consolidate_reactions=True,
			seed=120,
			epochs: int = 10,
			patience: int = 3,
			min_delta: float = 1e-4):

	# ----------------------------
	# Determine conditions
	# ----------------------------
	ks = (
		pd.unique(data[['relation', 'isi']].values.ravel('K'))
		if not consolidate_reactions
		else data["relation"].unique()
	)

	tokens = set(data["prime"]).union(set(data["target"]))
	emb_cache = load_or_build_embedding_cache(
		tokens,
		model=model,
		component=component,
		embedding_dir=embedding_dir
	)

	os.makedirs(graphs_dir, exist_ok=True)

	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.use_deterministic_algorithms(True)

	condition_test_accuracies = {}

	# ==========================================================
	# Loop over each condition
	# ==========================================================
	for k_condition in tqdm(ks, desc="Conditions"):
		if consolidate_reactions:
			subset = data[data["relation"] == k_condition]
			labels = subset["relation"].to_numpy()
		else:
			subset = data[
				(data["relation"] == k_condition[0]) &
				(data["isi"] == k_condition[1])
			]
			labels = list(zip(subset["relation"], subset["isi"]))

		if len(subset) < 5:
			continue

		primes = subset["prime"].apply(normalize_token).to_numpy()
		targets = subset["target"].apply(normalize_token).to_numpy()

		pts = list(zip(primes, targets))
		combined = list(zip(pts, labels))

		random.Random(seed).shuffle(combined)
		pts_shuffled, labels_shuffled = zip(*combined)

		observations = np.array([
			emb_cache[p] - emb_cache[t]
			for p, t in pts_shuffled
		])

		labels_encoded = to_categorical(labels_shuffled)
		num_classes = len(set(labels_shuffled))

		# ----------------------------
		# 80/10/10 split
		# ----------------------------
		xy = list(zip(observations, labels_encoded))

		train_split = int(0.8 * len(xy))
		val_split = train_split + int(0.1 * len(xy))

		train_obs = xy[:train_split]
		val_obs = xy[train_split:val_split]
		test_obs = xy[val_split:]

		architecture = np.linspace(
			observations[0].shape[0],
			num_classes,
			num=3,
			dtype=int
		)

		model_instance = k_model(architecture)
		optimizer = torch.optim.Adam(model_instance.parameters(), lr=0.001)
		criterion = nn.CrossEntropyLoss()

		train_losses = []
		val_losses = []

		best_val_loss = float("inf")
		best_model_state = None
		patience_counter = 0

		# ======================================================
		# Training loop
		# ======================================================
		epoch_iterator = tqdm(range(epochs),
							  desc=f"{k_condition} Epochs",
							  leave=False)

		for e in epoch_iterator:

			epoch_loss = 0.0
			epoch_val_loss = 0.0

			# ----- Validation -----
			model_instance.eval()
			with torch.no_grad():
				for x, y in tqdm(val_obs,
								 desc="Validation",
								 leave=False):
					x = torch.tensor(x, dtype=torch.float32)
					y = torch.tensor(y, dtype=torch.long)

					output = model_instance(x)
					loss = criterion(output.unsqueeze(0), y.unsqueeze(0))
					epoch_val_loss += loss.item()

			epoch_val_loss /= max(1, len(val_obs))

			# ----- Early stopping -----
			if epoch_val_loss < best_val_loss - min_delta:
				best_val_loss = epoch_val_loss
				best_model_state = model_instance.state_dict()
				patience_counter = 0
			else:
				patience_counter += 1
				if patience_counter >= patience:
					break

			# ----- Training -----
			model_instance.train()
			for x, y in tqdm(train_obs,
							 desc="Training",
							 leave=False):
				x = torch.tensor(x, dtype=torch.float32)
				y = torch.tensor(y, dtype=torch.long)

				output = model_instance(x)
				loss = criterion(output.unsqueeze(0), y.unsqueeze(0))

				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				epoch_loss += loss.item()

			epoch_loss /= len(train_obs)

			train_losses.append(epoch_loss)
			val_losses.append(epoch_val_loss)

			epoch_iterator.set_postfix(
				train_loss=f"{epoch_loss:.4f}",
				val_loss=f"{epoch_val_loss:.4f}"
			)

		# ----------------------------
		# Restore best model
		# ----------------------------
		if best_model_state is not None:
			model_instance.load_state_dict(best_model_state)

		# ----------------------------
		# Test evaluation
		# ----------------------------
		total_correct = 0
		total_samples = 0

		model_instance.eval()
		with torch.no_grad():
			for x, y in tqdm(test_obs,
							 desc="Testing",
							 leave=False):
				x = torch.tensor(x, dtype=torch.float32)
				y = torch.tensor(y, dtype=torch.long)

				output = model_instance(x)
				pred = torch.argmax(output).item()

				if pred == y.item():
					total_correct += 1
				total_samples += 1

		test_accuracy = total_correct / max(1, total_samples)
		condition_test_accuracies[str(k_condition)] = test_accuracy

		# ----------------------------
		# Save per-condition loss curve
		# ----------------------------
		plt.figure(figsize=(8, 5))
		epochs_range = range(1, len(train_losses) + 1)

		plt.plot(epochs_range, train_losses, marker="o", label="Train loss")
		plt.plot(epochs_range, val_losses, marker="o", label="Validation loss")

		plt.xlabel("Epoch")
		plt.ylabel("Cross-Entropy Loss")
		plt.title(f"{prefix}_{k_condition} Loss Curve")
		plt.legend()
		plt.grid(True)

		outpath = os.path.join(
			graphs_dir,
			f"{prefix}_{k_condition}_loss_curve.png"
		)

		plt.savefig(outpath, dpi=300, bbox_inches="tight")
		plt.close()

	# ==========================================================
	# Final Bar Chart
	# ==========================================================
	if condition_test_accuracies:

		conditions = list(condition_test_accuracies.keys())
		accuracies = list(condition_test_accuracies.values())

		plt.figure(figsize=(10, 6))
		plt.bar(conditions, accuracies)

		plt.xlabel("Condition")
		plt.ylabel("Test Accuracy")
		plt.title(f"{model}_{component} Test Accuracy per Condition")

		plt.xticks(rotation=45)
		plt.ylim(0, 1)
		plt.tight_layout()

		outpath = os.path.join(
			graphs_dir,
			f"{model}_{component}_testacc.png"
		)

		plt.savefig(outpath, dpi=300, bbox_inches="tight")
		plt.close()

		print("\nSaved overall test accuracy bar chart.")




if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str)
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
	prefix += f" {args.hf_id}_{args.component}"

	learn_k(
		data=data,
		model=args.hf_id,
		component=args.component,
		embedding_dir=args.embeddingdir,
		graphs_dir=args.graphs_dir,
		prefix=prefix,
		consolidate_reactions=not args.no_consolidate,
		seed=args.seed
	)