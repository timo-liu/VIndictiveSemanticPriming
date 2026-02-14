import pickle
import argparse
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Any, Callable
import os

top_k = 5

def latex_escape(s: str) -> str:
	return s.replace("_", "\\_")

def split_condition(cond: str):
	isi, name = cond.split("_", 1)
	return isi, name

def table_layer_tex(
		tlt: Dict[str, Dict[str, Counter]],
		all_layers: List[str],
		layer_sort_key: Callable[[str], Any],
		tex_path: str,
		collapse_tasks: bool = False,
		collapse_isis: bool = False
):
	layers = sorted(all_layers, key=layer_sort_key)
	tasks = sorted(tlt.keys())

	# collect universe of (isi, condition)
	isi_cond_pairs = set()
	for layer_counters in tlt.values():
		for counter in layer_counters.values():
			for c in counter:
				if '.' not in c:
					isi_cond_pairs.add(split_condition(c))

	with open(tex_path, "w") as f:

		# --------------------------------------------------
		# CASE 1: no collapse → task × isi × condition
		# --------------------------------------------------
		if not collapse_tasks and not collapse_isis:
			for task in tasks:
				for isi, cond in sorted(isi_cond_pairs):
					task_tex = latex_escape(task)
					isi_tex = latex_escape(isi)
					cond_tex = latex_escape(cond)

					f.write(f"% ---- {task_tex} | ISI {isi_tex} | {cond_tex} ----\n")
					f.write("\\begin{table}[H]\n\\centering\n")
					f.write(
						f"\\caption{{Layer counts — Task: {task_tex}, ISI: {isi_tex}, Condition: {cond_tex}}}\n"
					)

					f.write("\\begin{tabular}{lc}\n\\hline\n")
					f.write("Layer & Count \\\\\n\\hline\n")

					for layer in layers:
						key = f"{isi}_{cond}"
						count = tlt[task].get(layer, Counter()).get(key, 0)
						f.write(f"{latex_escape(layer)} & {count} \\\\\n")

					f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

		# --------------------------------------------------
		# CASE 2: collapse tasks → isi × condition
		# --------------------------------------------------
		elif collapse_tasks and not collapse_isis:
			for isi, cond in sorted(isi_cond_pairs):
				isi_tex = latex_escape(isi)
				cond_tex = latex_escape(cond)

				f.write(f"% ---- ISI {isi_tex} | {cond_tex} ----\n")
				f.write("\\begin{table}[H]\n\\centering\n")
				f.write(
					f"\\caption{{Layer counts collapsed across tasks — ISI: {isi_tex}, Condition: {cond_tex}}}\n"
				)

				f.write("\\begin{tabular}{lc}\n\\hline\n")
				f.write("Layer & Count \\\\\n\\hline\n")

				for layer in layers:
					key = f"{isi}_{cond}"
					count = sum(
						tlt[task].get(layer, Counter()).get(key, 0)
						for task in tasks
					)
					f.write(f"{latex_escape(layer)} & {count} \\\\\n")

				f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

		# --------------------------------------------------
		# CASE 3: collapse ISIs → task × condition
		# --------------------------------------------------
		elif collapse_isis and not collapse_tasks:
			conditions = sorted({cond for _, cond in isi_cond_pairs})

			for task in tasks:
				task_tex = latex_escape(task)

				for cond in conditions:
					cond_tex = latex_escape(cond)

					f.write(f"% ---- {task_tex} | {cond_tex} ----\n")
					f.write("\\begin{table}[H]\n\\centering\n")
					f.write(
						f"\\caption{{Layer counts collapsed across ISIs — Task: {task_tex}, Condition: {cond_tex}}}\n"
					)

					f.write("\\begin{tabular}{lc}\n\\hline\n")
					f.write("Layer & Count \\\\\n\\hline\n")

					for layer in layers:
						count = sum(
							tlt[task].get(layer, Counter()).get(f"{isi}_{cond}", 0)
							for isi, _ in isi_cond_pairs
							if _ == cond
						)
						f.write(f"{latex_escape(layer)} & {count} \\\\\n")

					f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")

		# --------------------------------------------------
		# CASE 4: collapse both → condition only
		# --------------------------------------------------
		else:
			conditions = sorted({cond for _, cond in isi_cond_pairs})

			for cond in conditions:
				cond_tex = latex_escape(cond)

				f.write(f"% ---- {cond_tex} (fully collapsed) ----\n")
				f.write("\\begin{table}[H]\n\\centering\n")
				f.write(
					f"\\caption{{Layer counts collapsed across tasks and ISIs — Condition: {cond_tex}}}\n"
				)

				f.write("\\begin{tabular}{lc}\n\\hline\n")
				f.write("Layer & Count \\\\\n\\hline\n")

				for layer in layers:
					count = sum(
						tlt[task].get(layer, Counter()).get(f"{isi}_{cond}", 0)
						for task in tasks
						for isi, _ in isi_cond_pairs
						if _ == cond
					)
					f.write(f"{latex_escape(layer)} & {count} \\\\\n")

				f.write("\\hline\n\\end{tabular}\n\\end{table}\n\n")


def most_salient_categories(
    tlt: Dict[str, Dict[str, Counter]],
    all_layers: List[str],
    layer_sort_key: Callable[[str], Any],
    tex_path: str,
    collapse_tasks: bool = False,
    collapse_isis: bool = False,
    k: int = 3,
):
    layers = sorted(all_layers, key=layer_sort_key)
    tasks = sorted(tlt.keys())

    # collect universe of (isi, condition)
    isi_cond_pairs = set()
    for layer_counters in tlt.values():
        for counter in layer_counters.values():
            for c in counter:
                if "." not in c:
                    isi_cond_pairs.add(split_condition(c))

    # --------------------------------------------------
    # build layer → Counter(category)
    # --------------------------------------------------
    layer_category_counts = defaultdict(Counter)

    for layer in layers:
        for task in tasks:
            counter = tlt[task].get(layer, Counter())

            for isi, cond in isi_cond_pairs:
                key = f"{isi}_{cond}"
                val = counter.get(key, 0)
                if val == 0:
                    continue

                # collapse ISIs
                if collapse_isis:
                    cat = cond
                else:
                    cat = key

                # collapse tasks handled by summing anyway
                layer_category_counts[layer][cat] += val

    # --------------------------------------------------
    # write single LaTeX table
    # --------------------------------------------------
    with open(tex_path, "w") as f:
        f.write("\\begin{table}[H]\n\\centering\n")
        f.write(
            f"\\caption{{Top-{k} most salient categories per layer"
            + (" (tasks collapsed)" if collapse_tasks else "")
            + (" (ISIs collapsed)" if collapse_isis else "")
            + "}}\n"
        )

        col_spec = "l" + "c" * k
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n\\hline\n")

        header = ["Layer"] + [f"Category {i+1}" for i in range(k)]
        f.write(" & ".join(header) + " \\\\\n\\hline\n")

        for layer in layers:
            topk = layer_category_counts[layer].most_common(k)
            cats = [latex_escape(cat) for cat, _ in topk]

            # pad if fewer than k
            while len(cats) < k:
                cats.append("")

            row = [latex_escape(layer)] + cats
            f.write(" & ".join(row) + " \\\\\n")

        f.write("\\hline\n\\end{tabular}\n\\end{table}\n")


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

	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"-i", "--input", required=False, help="Input file path", default="rsa_acc_whole.pkl"
	)
	arg_parser.add_argument("--collapse_tasks", action="store_true")
	arg_parser.add_argument("--collapse_isis", action="store_true")
	arg_parser.add_argument(
		"-o", "--output", required=False, help="Graphs go brrr", default="layer_rankings"
	)

	args = arg_parser.parse_args()
	os.makedirs(args.output, exist_ok=True)

	with open(args.input, "rb") as f:
		rsa_acc_whole = pickle.load(f)

	print(list(rsa_acc_whole["ldt"].keys())[0])

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

	table_layer_tex(
		layer_condition_tracker,
		all_layers,
		layer_sort_key,
		"layer_tex.tex",
		collapse_tasks=args.collapse_tasks,
		collapse_isis=args.collapse_isis
	)

	most_salient_categories(
		layer_condition_tracker,
		all_layers,
		layer_sort_key,
		"most_salient_layers.tex",
		collapse_tasks=args.collapse_tasks,
		collapse_isis=args.collapse_isis,
		k=3,  # top-k salient categories
	)
