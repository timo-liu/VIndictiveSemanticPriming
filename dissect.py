import pickle
import argparse
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Any, Callable
import os

top_k = 5

def layer_sort_key(layer):
	if layer == "word_embeddings":
		return -1
	if layer.startswith("encoder_layer_"):
		return int(layer.split("_")[-1])
	return float("inf")  # anything unexpected goes to the end


if __name__ == "__main__":
	# region pickle
	def leaf():
		return {"rhos": [], "ps": []}


	def level3():
		return defaultdict(leaf)


	def level2():
		return defaultdict(level3)


	def level1():
		return defaultdict(level2)
	# endregion pickle

	from typing import Dict, List, Any, Callable


	def table_layer_task_to_tex(
			tlt: Dict[str, Dict[str, Counter]],
			all_layers: List[str],
			layer_sort_key: Callable[[str], Any],
			tex_path: str
	):
		"""
		Generate a LaTeX table for each task and write to a .tex file.

		Parameters
		----------
		tlt : dict
			Mapping: task -> { layer -> Counter(condition -> count) }
		all_layers : list
			List of layer names.
		layer_sort_key : callable
			Sorting key for layers.
		tex_path : str
			Output .tex file path.
		"""

		layers = sorted(all_layers, key=layer_sort_key)

		with open(tex_path, "w") as f:
			for task, condition_counters in tlt.items():

				# Collect all conditions across all layers for this task
				all_conditions = set()
				for counter in condition_counters.values():
					all_conditions.update(counter.keys())
				all_conditions = sorted(all_conditions)

				# Begin table
				f.write(f"% ---- Task: {task} ----\n")
				f.write("\\begin{table}[h!]\n")
				f.write("\\centering\n")
				f.write(f"\\caption{{Results for task: {task}}}\n")

				# Column format: 1 left column + N centered columns
				col_format = "l" + "c" * len(all_conditions)
				f.write(f"\\begin{{tabular}}{{{col_format}}}\n")
				f.write("\\hline\n")

				# Header row
				header = ["Layer"] + all_conditions
				f.write(" & ".join(header) + " \\\\\n")
				f.write("\\hline\n")

				# Data rows
				for layer in layers:
					counter = condition_counters.get(layer, Counter())
					row = [layer] + [str(counter.get(cond, 0)) for cond in all_conditions]
					f.write(" & ".join(row) + " \\\\\n")

				f.write("\\hline\n")
				f.write("\\end{tabular}\n")
				f.write("\\end{table}\n\n")


	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"-i", "--input", required=False, help="Input file path", default="rsa_acc_whole.pkl"
	)
	arg_parser.add_argument(
		"-o", "--output", required=False, help="Graphs go brrr", default="layer_rankings"
	)

	args = arg_parser.parse_args()
	os.makedirs(args.output, exist_ok=True)

	with open(args.input, "rb") as f:
		rsa_acc_whole = pickle.load(f)

	top3_layers = defaultdict(lambda: defaultdict(dict))
	all_layers = set()

	# store dict of [layer] = Counter( conditions)
	layer_condition_tracker = {
		"nam" : defaultdict(Counter),
		"ldt": defaultdict(Counter)
	}

	for task, condition_data in rsa_acc_whole.items():
		for condition, layer_data in condition_data.items():
			per_model = defaultdict(list)
			for model, layer_data in layer_data.items():
				if "distil" in model:
					continue
				for layer, rho_data in layer_data.items():
					all_layers.add(layer)
					rizz = [1 - r for r in rho_data["rhos"]]
					rho_mean = np.mean(rizz)
					error = st.sem(rizz)

					per_model[model].append((layer, rho_mean, error))

			for model, layers in per_model.items():
				layers = sorted(layers, key=lambda x: x[1])  # lower is better
				top3_layers[condition][task][model] = {layer for layer, _, _ in layers[:top_k]}
				for l, _, _ in layers[:top_k]:
					layer_condition_tracker[task][l][condition] += 1

	condition_task_layer_counts = defaultdict(lambda: defaultdict(Counter))

	for condition, task_data in top3_layers.items():
		for task, model_data in task_data.items():
			for layerset in model_data.values():
				condition_task_layer_counts[condition][task].update(layerset)

	condition_layer_counts = defaultdict(Counter)

	for condition, task_data in top3_layers.items():

		model_union = defaultdict(set)

		for task, model_data in task_data.items():
			for model, layerset in model_data.items():
				model_union[model].update(layerset)

		for layerset in model_union.values():
			condition_layer_counts[condition].update(layerset)

	# 4. Plot aggregated per condition
	for condition, layer_counter in condition_layer_counts.items():
		layers = sorted(all_layers, key=layer_sort_key)
		counts = [layer_counter.get(l, 0) for l in layers]

		plt.figure()
		plt.bar(layers, counts)
		plt.xlabel("Layer")
		plt.ylabel("Number of models")
		plt.title(f"Top-{top_k} Layer Frequency — {condition}")
		plt.xticks(rotation=45, ha="right", fontsize=8)
		plt.tight_layout()

		save_path = os.path.join(args.output, f"top{top_k}_layer_frequency_{condition}.png")
		plt.savefig(save_path, dpi=300)
		plt.close()

	for condition, task_data in condition_task_layer_counts.items():
		for task, layer_counter in task_data.items():
			layers = sorted(all_layers, key=layer_sort_key)
			counts = [layer_counter.get(l, 0) for l in layers]

			plt.figure()
			plt.bar(layers, counts)
			plt.xlabel("Layer")
			plt.ylabel("Number of models")
			plt.title(f"{task} — {condition}")
			plt.xticks(rotation=45, ha="right", fontsize=8)
			plt.tight_layout()

			save_path = os.path.join(args.output, f"top{top_k}_layer_frequency_{task}_{condition}.png")
			plt.savefig(save_path, dpi=300)
			plt.close()

	table_layer_task_to_tex(layer_condition_tracker, all_layers, layer_sort_key, "layer_tex.tex")
