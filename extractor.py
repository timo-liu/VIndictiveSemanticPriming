from Definitions.GutsGorer import GutsGorer
from Definitions.Utils import *
import pandas as pd
import argparse
from scipy.stats import spearmanr
import os
import torch
import json


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def write_rho(
	rho_file,
	*,
	model,
	component,
	relation,
	isi,
	dataset,
	rho,
	pvalue,
	n,
	bs_index=None
):

	if bs_index is None:
		entry_key = f"{dataset}|{model}|{component}|{relation}|isi={isi}"
	else:
		entry_key = f"{dataset}|{model}|{component}|{relation}|isi={isi}"

	record = {
		"dataset": str(dataset),
		"model": str(model),
		"component": str(component),
		"relation": str(relation),
		"isi": int(isi),
		"spearman_rho": float(rho),
		"p_value": float(pvalue),
		"n": int(n),
		"bs": int(bs_index) if bs_index is not None else None
	}

	if os.path.exists(rho_file):
		with open(rho_file, "r") as f:
			try:
				data = json.load(f)
			except json.JSONDecodeError:
				data = {}
	else:
		data = {}

	data[entry_key] = record

	with open(rho_file, "w") as f:
		json.dump(data, f, indent=2)


# ---------------------------------------------------------------------
# Cosine computation (lazy model loading)
# ---------------------------------------------------------------------

def compute_or_load_cosines(
	data,
	output_dir,
	*,
	model,
	component,
	embedding_dir="D:/heat_embeddings",
	bs_index=0
):
	data = data[data["bs"] == bs_index].copy()
	os.makedirs(output_dir, exist_ok=True)
	os.makedirs(embedding_dir, exist_ok=True)

	output_file = os.path.join(
		output_dir,
		f"cosines_{model.replace('/', '_')}_{component}.csv"
	)

	if os.path.exists(output_file):
		return pd.read_csv(output_file)

	print("Cosine file not found. Loading model...")
	gg = GutsGorer(model)

	cache_path = os.path.join(
		embedding_dir,
		f"{model.replace('/', '_')}.npz"
	)

	emb_cache = load_embedding_cache(cache_path)
	new_records = []

	for _, row in data.iterrows():

		if pd.notna(row[["prime", "target"]]).all():

			prime = str(row["prime"])
			target = str(row["target"])

			if component == "word_embeddings":
				prime_vec = gg.compute_embedding(prime, component="word_embeddings")
				target_vec = gg.compute_embedding(target, component="word_embeddings")

			elif component.startswith("encoder_layer_"):
				layer = int(component.split("_")[-1])
				prime_layers = get_or_compute_encoder_layers(prime, gg, emb_cache, component)
				target_layers = get_or_compute_encoder_layers(target, gg, emb_cache, component)
				prime_vec = prime_layers[layer]
				target_vec = target_layers[layer]
			else:
				raise ValueError(f"Unknown component: {component}")

			cosine_val = gg.cosine_similarity(
				torch.from_numpy(prime_vec),
				torch.from_numpy(target_vec)
			)

			new_records.append({
				"cosine": cosine_val,
				"RT": row["RT"],
				"isi": row["isi"],
				"relation": row["relation"]
			})

	save_embedding_cache(emb_cache, cache_path)

	df_new = pd.DataFrame(new_records)
	df_new.to_csv(output_file, index=False)

	return df_new


# ---------------------------------------------------------------------
# Correlation computation (single file)
# ---------------------------------------------------------------------

def compute_correlations(
	df,
	*,
	model,
	component,
	dataset,
	rhos_file,
	bs_index=None
):

	relations = df["relation"].unique()
	isis = df["isi"].unique()

	for rel in relations:
		for isi in isis:

			subset = df[
				(df["relation"] == rel) &
				(df["isi"] == isi)
			]

			if len(subset) < 3:
				continue

			res = spearmanr(subset["cosine"], subset["RT"])

			write_rho(
				rhos_file,
				model=model,
				component=component,
				relation=rel,
				isi=isi,
				dataset=dataset,
				rho=res.correlation,
				pvalue=res.pvalue,
				n=len(subset),
				bs_index=bs_index
			)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('data', type=str)
	parser.add_argument('-i', '--hf_id', default="bert-base-uncased")
	parser.add_argument('-c', '--component', required=True)
	parser.add_argument('-p', '--condition', required=True)
	parser.add_argument('-r', '--rhos', required=True)
	parser.add_argument('--cache', required=True)
	parser.add_argument('--bs', type=int, help="num bootstraps")

	args = parser.parse_args()

	# --------------------------------------------
	# If bootstrap index provided → load that file
	# --------------------------------------------
	data_path = args.data

	print(f"Loading data: {data_path}")

	data = parse_data(
		data_path,
		extract_dict={
			"prime": "prime",
			"target": "target",
			"relation": "relation",
			"RT": "RT",
			"isi": "isi",
			"bs" : "bs"
		}
	)

	for i in tqdm(range(args.bs), desc="Processing bootstraps"):
		df_bs = compute_or_load_cosines(
			data,
			args.cache,
			component=args.component,
			model=args.hf_id,
			bs_index=i
		)

		if df_bs.empty:
			print(f"No data for bootstrap {i}, skipping")
			continue

		compute_correlations(
			df_bs,
			model=args.hf_id,
			component=args.component,
			dataset=args.condition,
			rhos_file=args.rhos,
			bs_index=i
		)
