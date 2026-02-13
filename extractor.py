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
	n
):
	"""
	Append/update Spearman results in a shared JSON file.
	Keys are deterministic, so reruns overwrite cleanly.
	"""

	entry_key = f"{dataset}|{model}|{component}|{relation}|isi={isi}"

	record = {
		"dataset": dataset,
		"model": model,
		"component": component,
		"relation": relation,
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
    model="bert-base-uncased",
    component="",
    embedding_dir="D:/redone_heatmaps"
):
    """
    Compute cosine similarities using cached encoder-layer embeddings.
    Embeddings are cached per token only, independent of relation or condition.
    """

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(embedding_dir, exist_ok=True)
    cache_path = os.path.join(embedding_dir, f"{model.replace('/', '_')}.npz")

    # Load token-layer cache
    emb_cache = load_embedding_cache(cache_path)

    output_file = os.path.join(
        output_dir,
        f"cosines_{model.replace('/', '_')}_{component}.csv"
    )

    if os.path.exists(output_file):
        df_saved = pd.read_csv(output_file)
    else:
        df_saved = pd.DataFrame(columns=["cosine", "RT", "isi", "relation"])

    new_records = []

    for _, row in data.iterrows():
        if pd.notna(row[["prime", "target"]]).all() and str(row["prime"]).isalpha() and str(row["target"]).isalpha():
            prime = str(row["prime"])
            target = str(row["target"])

            # Compute embeddings
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
                "relation": row["relation"]  # just store for downstream plotting
            })

    # Save updated cache
    save_embedding_cache(emb_cache, cache_path)

    if new_records:
        df_new = pd.DataFrame(new_records)
        df_saved = pd.concat([df_saved, df_new], ignore_index=True)
        df_saved.to_csv(output_file, index=False)

    return df_saved


# ---------------------------------------------------------------------
# Plotting (relation-based)
# ---------------------------------------------------------------------

def plot_relation(
	df,
	relation,
	*,
	output_prefix="cosine_vs_RT",
	component="",
	model="bert-base-uncased",
	graphs="graphs",
	isi=50,
	suffix=""
):
	if not os.path.exists(graphs):
		os.makedirs(graphs)

	output_folder = os.path.join(graphs, model.replace("/", "_"))
	os.makedirs(output_folder, exist_ok=True)

	subset = df[
		(df["relation"] == relation) &
		(df["isi"] == isi)
	]

	if len(subset) < 3:
		return

	cosines = subset["cosine"].tolist()
	RTs = subset["RT"].tolist()

	res = spearmanr(cosines, RTs)

	write_rho(
		RHOS,
		model=model,
		component=component,
		relation=relation,
		isi=isi,
		dataset=suffix,
		rho=res.correlation,
		pvalue=res.pvalue,
		n=len(cosines)
	)

	plt.scatter(cosines, RTs, alpha=0.7)
	plt.xlabel("Cosine similarity")
	plt.ylabel("Reaction time (RT)")
	plt.title(f"{relation} | {component} | isi={isi}")
	plt.gca().invert_xaxis()

	result = linregress(cosines, RTs)
	x = np.unique(cosines)
	plt.plot(x, result.slope * x + result.intercept, color="red")

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
	plt.savefig(
		os.path.join(
			output_folder,
			f"{output_prefix}_{relation}_{component}_{isi}_{suffix}.png"
		)
	)
	plt.close()

def plot_everything_lmao(
	df,
	*,
	output_prefix="AGGREGATED_cosine_vs_RT",
	component="",
	model="bert-base-uncased",
	graphs="graphs",
	isi=50,
	suffix=""
):
	if not os.path.exists(graphs):
		os.makedirs(graphs)

	output_folder = os.path.join(graphs, model.replace("/", "_"))
	os.makedirs(output_folder, exist_ok=True)

	subset = df[
		(df["isi"] == isi)
	]

	if len(subset) < 3:
		return

	cosines = subset["cosine"].tolist()
	RTs = subset["RT"].tolist()

	res = spearmanr(cosines, RTs)

	write_rho(
		RHOS,
		model=model,
		component=component,
		relation="aggregated",
		isi=isi,
		dataset=suffix,
		rho=res.correlation,
		pvalue=res.pvalue,
		n=len(cosines)
	)

	plt.scatter(cosines, RTs, alpha=0.7)
	plt.xlabel("Cosine similarity")
	plt.ylabel("Reaction time (RT)")
	plt.title(f"Aggregated conditions | {component} | isi={isi}")
	plt.gca().invert_xaxis()

	result = linregress(cosines, RTs)
	x = np.unique(cosines)
	plt.plot(x, result.slope * x + result.intercept, color="red")

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
	plt.savefig(
		os.path.join(
			output_folder,
			f"{output_prefix}_AGGREGATED_{component}_{isi}_{suffix}.png"
		)
	)
	plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data', type=str, help='data file')
	arg_parser.add_argument('-i', '--hf_id', default="bert-base-uncased")
	arg_parser.add_argument('-s', '--save', default="data")
	arg_parser.add_argument('-c', '--component', default="")
	arg_parser.add_argument('-n', '--n_data', type=int, default=5000)
	arg_parser.add_argument('-g', '--graphs', default="graphs")
	arg_parser.add_argument('-p', '--condition', required=True, help="Dataset label")
	arg_parser.add_argument('-m', '--aggregate', action = "store_true")
	args = arg_parser.parse_args()

	gg = GutsGorer(args.hf_id)

	data = parse_data(
		args.data,
		extract_dict={
			"prime": "prime",
			"target": "target",
			"relation" : "relation",
			"RT": "RT",
			"isi": "isi"
		}
	)


	df_all = compute_or_load_cosines(
		data,
		gg,
		args.save,
		component=args.component,
		model=args.hf_id
	)
	relations = df_all["relation"].unique()
	isis = df_all["isi"].unique()

	print(f"Found relations: {relations}")

	if not args.aggregate:
		for rel in relations:
			for isi in isis:
				plot_relation(
					df_all,
					rel,
					component=args.component,
					model=args.hf_id,
					graphs=args.graphs,
					isi=isi,
					suffix=args.condition
				)
				print(f"Saved plot for relation {rel}, isi {isi}")
	else:
		for isi in isis:
			plot_everything_lmao(
				df_all,
				component=args.component,
				model=args.hf_id,
				graphs=args.graphs,
				isi=isi,
				suffix=args.condition
			)