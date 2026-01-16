"""
Trying to train a model to predict the reaction time from the embedding.
Might be fun, might be cool.
"""
from Definitions.RTPredictor import RTPredictor
from Definitions.GutsGorer import GutsGorer
from typing import Tuple
from Definitions.Utils import *
import argparse
import torch
import numpy as np
import os
import random
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import linregress

def get_or_compute_embeddings(data, gg, component: str, hf_id: str, cache_dir="data_cache", max_data: int = 2000):
	"""
	Returns a DataFrame with columns:
		['prime', 'target', 'prime_embedding', 'target_embedding', 'embedding', 'RT']
	- prime_embedding / target_embedding: individual embeddings (NumPy arrays)
	- embedding: concatenation of prime + target embeddings
	Caches via pickle to avoid recomputation
	"""

	os.makedirs(cache_dir, exist_ok=True)
	cache_file = os.path.join(cache_dir, f"embeddings_{hf_id}_{component}.pkl")

	# Load cached embeddings if available
	if os.path.exists(cache_file):
		df_cached = pd.read_pickle(cache_file)
		print(f"Loaded cached embeddings: {len(df_cached)} rows from {cache_file}")
	else:
		df_cached = pd.DataFrame(columns=['prime', 'target', 'prime_embedding', 'target_embedding', 'embedding', 'RT', 'isi'])

	# Build a fast lookup for already computed embeddings
	prime_emb_cache = {}
	target_emb_cache = {}

	# Fill caches from cached DataFrame
	for row in df_cached.itertuples():
		prime_emb_cache[row.prime] = row.prime_embedding
		target_emb_cache[row.target] = row.target_embedding

	new_records = []
	for _, row in data.iterrows():
		prime, target = str(row['prime']), str(row['target'])
		RT = row['RT']
		isi = row['isi']

		# Skip if already cached as full row
		cached_row = df_cached[
			(df_cached['prime'] == prime) &
			(df_cached['target'] == target)
		]
		if not cached_row.empty:
			new_records.append(cached_row.iloc[0].to_dict())
			continue

		# Compute embeddings individually
		if prime not in prime_emb_cache:
			prime_emb_cache[prime] = gg.compute_embedding(prime, component)
		if target not in target_emb_cache:
			target_emb_cache[target] = gg.compute_embedding(target, component)

		prime_emb = prime_emb_cache[prime]
		target_emb = target_emb_cache[target]
		emb_concat = np.concatenate([prime_emb, target_emb], axis=0)

		new_records.append({
			'prime': prime,
			'target': target,
			'prime_embedding': prime_emb,
			'target_embedding': target_emb,
			'embedding': emb_concat,
			'RT': RT,
			'isi': isi
		})

		if len(new_records) >= max_data:
			break

	# Combine with cached DataFrame
	if new_records:
		df_new = pd.DataFrame(new_records)
		df_combined = pd.concat([df_cached, df_new], ignore_index=True)
		df_combined.to_pickle(cache_file)
		print(f"Updated cache with {len(df_combined)} rows → {cache_file}")
	else:
		df_combined = df_cached
		print("No new embeddings to compute.")

	return df_combined

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data', type=str, help='data file')
	arg_parser.add_argument('--epochs', type=int, default=10)
	arg_parser.add_argument('--patience', type=int, default=2)
	arg_parser.add_argument('--batch_size', type=int, default=128)
	arg_parser.add_argument('--seed', type=int, default=123)
	arg_parser.add_argument('--architecture', type=Tuple[int, int], nargs='+', default=[(768*2, 384), (384, 96), (96, 24),(24,1) ])
	arg_parser.add_argument('--data_size', type=int, default=3000)
	arg_parser.add_argument('--hf_id', type=str, default="bert-base-uncased")
	arg_parser.add_argument('-c','--component', type=str, default="", help="Which component of the embedding to use")
	arg_parser.add_argument('--primecondition', type=int, default=1)
	arg_parser.add_argument('--graphs', type=str, default="rtgraphs")
	args = arg_parser.parse_args()

	assert os.path.exists(args.data)

	graphs_dir = os.path.join(args.graphs, f"{args.hf_id}_{args.component}")
	os.makedirs(graphs_dir, exist_ok=True)

	torch.manual_seed(args.seed)
	random.seed(args.seed)
	np.random.seed(args.seed)

	data = parse_data(args.data)
	data = data.sample(frac=1.0, random_state=args.seed)

	# drop rows that have na in RT
	data = data.dropna(subset=['RT'])
	data = data[data["primecondition"].astype(int) == args.primecondition]

	gg = GutsGorer(args.hf_id)

	df_embeddings = get_or_compute_embeddings(data, gg, component=args.component, hf_id=args.hf_id, max_data=args.data_size)

	# Prepare training tensors
	X = torch.tensor(df_embeddings['embedding'].tolist(), dtype=torch.float32)
	prime_X = torch.stack([torch.tensor(e, dtype=torch.float32) for e in df_embeddings['prime_embedding']])
	target_X = torch.stack([torch.tensor(e, dtype=torch.float32) for e in df_embeddings['target_embedding']])
	y = torch.tensor(df_embeddings['RT'].tolist(), dtype=torch.float32).unsqueeze(1)

	y_mean = y.mean()
	y_std = y.std()
	y = (y - y_mean) / y_std

	N = X.size(0)
	perm = torch.randperm(N)
	X = X[perm]
	prime_X = prime_X[perm]
	target_X = target_X[perm]
	y = y[perm]

	n_train = int(0.8 * N)
	n_val = int(0.1 * N)

	# ===== Slices =====
	xtr = X[:n_train]
	ytr = y[:n_train]

	xval = X[n_train:n_train + n_val]
	yval = y[n_train:n_train + n_val]

	xtest = X[n_train + n_val:]
	ytest = y[n_train + n_val:]

	# Also keep prime/target embeddings for test set
	xtest_prime = prime_X[n_train + n_val:]
	xtest_target = target_X[n_train + n_val:]

	criterion = torch.nn.MSELoss
	activation = torch.nn.ReLU
	model = RTPredictor(args.architecture, activation, criterion)
	model.train(
		xtr=xtr,
		ytr=ytr,
		xval=xval,
		yval=yval,
		epochs=args.epochs,
		batch_size=args.batch_size,
		patience=args.patience,
		project_name=f"{args.hf_id}_{args.component}_{args.primecondition}"
	)

	test_loss, preds = model.test_model(
		xtest=xtest,
		ytest=ytest,
		batch_size=args.batch_size,
		return_preds=True
	)
	cos_sims = [gg.cosine_similarity(p, t) for p,t in zip(xtest_prime, xtest_target)]
	original_rts = ytest * y_std + y_mean
	original_rts = original_rts.squeeze().tolist()
	preds = preds * y_std + y_mean
	cos_sims = np.array(cos_sims)
	original_rts = np.array(original_rts)
	preds_denorm = np.array(preds).squeeze().tolist()

	# Plot setup
	plt.figure(figsize=(6, 4))

	# Scatter original RTs
	plt.scatter(cos_sims, original_rts, alpha=0.6, label='Original RTs', color='blue')

	# Scatter predicted RTs
	plt.scatter(cos_sims, preds_denorm, alpha=0.6, label='Predicted RTs', color='orange')

	plt.xlabel("Cosine similarity")
	plt.ylabel("Reaction time (RT)")
	plt.title(f"Cosine similarity vs RT (primecondition={args.primecondition}, component={args.component})")
	plt.gca().invert_xaxis()  # if you want to match your previous plots

	# ===== Regression for original RTs =====
	res_orig = linregress(cos_sims, original_rts)
	x_unique = np.linspace(cos_sims.min(), cos_sims.max(), 100)
	plt.plot(x_unique, res_orig.slope * x_unique + res_orig.intercept, color='blue', linestyle='--',
			 label='Orig RTs fit')
	plt.text(0.05, 0.95,
			 f"Orig slope={res_orig.slope:.3g}\nR²={res_orig.rvalue ** 2:.3g}\np={res_orig.pvalue:.3g}",
			 transform=plt.gca().transAxes, verticalalignment="top", color='blue')

	# ===== Regression for predicted RTs =====

	res_pred = linregress(cos_sims, preds_denorm)
	plt.plot(x_unique, res_pred.slope * x_unique + res_pred.intercept, color='orange', linestyle='--',
			 label='Pred RTs fit')
	plt.text(0.05, 0.80,
			 f"Pred slope={res_pred.slope:.3g}\nR²={res_pred.rvalue ** 2:.3g}\np={res_pred.pvalue:.3g}",
			 transform=plt.gca().transAxes, verticalalignment="top", color='orange')

	plt.legend()
	plt.tight_layout()
	plt.savefig(os.path.join(graphs_dir, f"cosine_vs_RT_{args.hf_id}_{args.component}_{args.primecondition}.png"))
	plt.close()