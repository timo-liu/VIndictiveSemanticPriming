from Definitions.GutsGorer import GutsGorer
from Definitions.Utils import *
import pandas as pd
import argparse
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress, spearmanr
import os
import torch
import json

RHOS = "rhos.json"

def write_rho(
	rho_file,
	*,
	model,
	component,
	primecondition,
	isi,
	dataset,
	rho,
	pvalue,
	n
):
	"""
	Append/update Spearman results in a shared JSON file.
	Keys are deterministic, so reruns overwrite cleanly.
	"""

	entry_key = f"{dataset}|{model}|{component}|{primecondition}|isi={isi}"

	record = {
		"dataset": dataset,
		"model": model,
		"component": component,
		"primecondition": primecondition,
		"isi": isi,
		"spearman_rho": float(rho),
		"p_value": float(pvalue),
		"n": int(n)
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


def compute_or_load_cosines(
	data,
	gg,
	output_dir,
	*,
	n=200,
	model="bert-base-uncased",
	component="",
	embedding_dir="embeddings"
):
	"""
	Compute cosine similarities using cached encoder-layer embeddings.
	- word_embeddings: computed on the fly (single vector)
	- encoder_layer_X: cached per word as (layers, dim)
	"""

	os.makedirs(output_dir, exist_ok=True)
	cache_path = os.path.join(embedding_dir, f"{model.replace('/', '_')}.npz")

	# load encoder-layer cache
	emb_cache = load_embedding_cache(cache_path)

	output_file = os.path.join(
		output_dir,
		f"cosines_{model.replace('/', '_')}_{component}.csv"
	)

	if os.path.exists(output_file):
		df_saved = pd.read_csv(output_file)
	else:
		df_saved = pd.DataFrame(columns=["cosine", "RT", "isi", "primecondition"])

	primeconditions = data["primecondition"].unique()
	new_records = []

	for cond in primeconditions:
		saved_count = len(df_saved[df_saved["primecondition"] == cond])
		to_process = max(0, n - saved_count)
		if to_process == 0:
			continue

		subset = data[data["primecondition"] == cond]
		added = 0

		for _, line in subset.iterrows():
			if added >= to_process:
				break

			if (
				pd.notna(line[["prime", "target"]]).all()
				and str(line["prime"]).isalpha()
				and str(line["target"]).isalpha()
			):
				prime = str(line["prime"])
				target = str(line["target"])

				# ---- COMPONENT SELECTION ----
				if component == "word_embeddings":
					prime_vec = gg.compute_embedding(
						prime, component="word_embeddings"
					)
					target_vec = gg.compute_embedding(
						target, component="word_embeddings"
					)

				elif component.startswith("encoder_layer_"):
					layer = int(component.split("_")[-1])

					prime_layers = get_or_compute_encoder_layers(
						prime, gg, emb_cache, component
					)
					target_layers = get_or_compute_encoder_layers(
						target, gg, emb_cache, component
					)

					prime_vec = prime_layers[layer]
					target_vec = target_layers[layer]

				else:
					raise ValueError(f"Unknown component: {component}")

				new_records.append({
					"cosine": gg.cosine_similarity(
						torch.from_numpy(prime_vec),
						torch.from_numpy(target_vec)
					),
					"RT": line["RT"],
					"isi": line["isi"],
					"primecondition": cond
				})

				added += 1

	# save updated encoder-layer cache
	save_embedding_cache(emb_cache, cache_path)

	if new_records:
		df_new = pd.DataFrame(new_records)
		df_saved = pd.concat([df_saved, df_new], ignore_index=True)
		df_saved.to_csv(output_file, index=False)

	return df_saved


def plot_condition(df, condition, output_prefix : str ="cosine_vs_RT", component : str = "", model :str = "bert-base-uncased", graphs : str = "graphs", isi=50, suffix : str = ""):
	"""Scatter + regression plot for a single primecondition"""

	if not os.path.exists(graphs):
		os.makedirs(graphs)

	output_folder = os.path.join(graphs, model.replace("/","_"))
	if not os.path.exists(output_folder):
		os.makedirs(output_folder)

	subset = df[df['primecondition'] == condition]
	subset = subset[subset['isi'] == isi]
	cosines = subset['cosine'].tolist()
	RTs = subset['RT'].tolist()

	res = spearmanr(cosines, RTs)

	write_rho(
		RHOS,
		model=model,
		component=component,
		primecondition=condition,
		isi=isi,
		dataset=suffix,
		rho=res.correlation,
		pvalue=res.pvalue,
		n=len(cosines)
	)

	plt.scatter(cosines, RTs, alpha=0.7)
	plt.xlabel("Cosine similarity")
	plt.ylabel("Reaction time (RT)")
	plt.title(f"Cosine similarity vs RT (primecondition={condition}, component={component})")
	plt.gca().invert_xaxis()

	# Regression line
	result = linregress(cosines, RTs)
	x = np.unique(cosines)
	plt.plot(x, result.slope * x + result.intercept, color='red')

	# Annotate p-value
	plt.text(
		0.05, 0.95,
		(
			f"Linear:\n"
			f"  slope = {result.slope:.3g}\n"
			f"  R² = {result.rvalue ** 2:.3g}\n"
			f"  p = {result.pvalue:.3g}\n\n"
			f"Spearman:\n"
			f"  rho = {res.correlation:.3g}\n"
			f"  p = {res.pvalue:.3g}"
		),
		transform=plt.gca().transAxes,
		verticalalignment="top"
	)

	plt.tight_layout()
	plt.savefig(os.path.join(output_folder, f"{output_prefix}_condition_{condition}_{model.replace("/", "_")}_{component}_{isi}_{suffix}.png"))
	plt.close()

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data', type=str, help='data file')
	arg_parser.add_argument('-i', '--hf_id', required=False, type=str, help='hf id', default="bert-base-uncased")
	arg_parser.add_argument('-s', '--save', required=False, type=str, help='output dir', default="data")
	arg_parser.add_argument('-c', '--component', required=False, type=str, help='component of interest', default="")
	arg_parser.add_argument('-n', '--n_data', required=False, type=int, help='number of data points', default=5000)
	arg_parser.add_argument('-g', '--graphs', required=False, type=str, help='graphs dir', default="graphs")
	arg_parser.add_argument('-p', "--condition", required=True, type=str, help="Naming task or ldt")
	args = arg_parser.parse_args()

	gg = GutsGorer(args.hf_id)
	data = parse_data(args.data)
	data = data[data["accuracy"] == 1]
	data = data.groupby(['prime', 'target', 'isi', 'primecondition'], as_index=False)['RT'].mean()
	df_all = compute_or_load_cosines(data, gg, args.save, n=args.n_data, component=args.component, model=args.hf_id)

	# Get unique primeconditions
	conditions = df_all['primecondition'].unique()
	isis = df_all['isi'].unique()
	print(f"Found primeconditions: {conditions}")

	# Generate plots per condition
	for cond in conditions:
		for isi in isis:
			plot_condition(df_all, cond, component=args.component, model=args.hf_id, graphs=args.graphs, isi=isi, suffix=args.condition)
			print(f"Saved plot for primecondition {cond}, isi {isi}")