import pickle
import argparse
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
from typing import Dict, List, Any, Callable, Tuple
import os
import trimesh

top_k = 5


def latex_escape(s: str) -> str:
	return s.replace("_", "\\_")

def split_condition(cond: str):
	isi, name = cond.split("_", 1)
	return isi, name

def layer_condition_matrix(
	tlt: Dict[str, Dict[str, Counter]],
	all_layers: List[str],
	layer_sort_key: Callable[[str], Any],
	collapse_tasks: bool = False,
	collapse_isis: bool = False,
) -> Tuple[np.ndarray, List[str], List[str]]:
	"""
	Returns:
		counts: (n_layers, n_conditions) integer matrix
		layers: ordered layer labels
		conditions: ordered condition labels
	"""

	layers = sorted(all_layers, key=layer_sort_key)
	tasks = sorted(tlt.keys())

	# --------------------------------------------------
	# Collect universe of (isi, condition)
	# --------------------------------------------------
	isi_cond_pairs = set()
	for layer_counters in tlt.values():
		for counter in layer_counters.values():
			for c in counter:
				if '.' not in c:
					isi_cond_pairs.add(split_condition(c))

	# --------------------------------------------------
	# Decide condition axis after collapsing
	# --------------------------------------------------
	if collapse_isis:
		conditions = sorted({cond for _, cond in isi_cond_pairs})
		condition_keys = conditions
	else:
		condition_keys = sorted(
			f"{isi}_{cond}" for isi, cond in isi_cond_pairs
		)
		conditions = condition_keys

	counts = np.zeros((len(layers), len(condition_keys)), dtype=int)

	# --------------------------------------------------
	# Fill matrix
	# --------------------------------------------------
	for i, layer in enumerate(layers):
		for j, key in enumerate(condition_keys):

			if collapse_tasks and collapse_isis:
				# sum over tasks and ISIs → condition only
				cond = key
				counts[i, j] = sum(
					tlt[task].get(layer, Counter()).get(f"{isi}_{cond}", 0)
					for task in tasks
					for isi, _ in isi_cond_pairs
					if _ == cond
				)

			elif collapse_tasks:
				# sum over tasks → isi × condition
				counts[i, j] = sum(
					tlt[task].get(layer, Counter()).get(key, 0)
					for task in tasks
				)

			elif collapse_isis:
				# sum over ISIs → condition only (per task already merged)
				cond = key
				counts[i, j] = sum(
					tlt[task].get(layer, Counter()).get(f"{isi}_{cond}", 0)
					for task in tasks
					for isi, _ in isi_cond_pairs
					if _ == cond
				)

			else:
				raise ValueError(
					"No-collapse mode yields task-specific matrices; "
					"handle per-task upstream."
				)

	return counts, layers, conditions


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

	counts, layers, conditions = layer_condition_matrix(
		layer_condition_tracker,
		all_layers,
		layer_sort_key,
		collapse_tasks=True,
		collapse_isis=True,
	)


	def normalize_z(z: np.ndarray) -> np.ndarray:
		"""
		Normalize array to the range [0, 1].

		If all values are equal, returns an array of zeros.
		"""
		z = np.asarray(z, dtype=float)
		z_min = z.min()
		z_max = z.max()

		if z_max == z_min:
			return np.zeros_like(z)

		return (z - z_min) / (z_max - z_min)

	def heightmap_to_column_mesh(
			Z: np.ndarray,
			cell_size: float = 1.0,  # cm
			base_thickness: float = 1.0  # cm
	) -> trimesh.Trimesh:
		"""
		Convert a 2D heightmap into a solid mesh made of square columns.

		Each Z[i, j] becomes a 1×1×Z column on top of a shared base.
		"""

		rows, cols = Z.shape
		meshes = []

		# --------------------------------------------------
		# Base plate
		# --------------------------------------------------
		base = trimesh.creation.box(
			extents=(
				cols * cell_size,
				rows * cell_size,
				base_thickness,
			)
		)
		base.apply_translation(
			(
				cols * cell_size / 2,
				rows * cell_size / 2,
				base_thickness / 2,
			)
		)
		meshes.append(base)

		# --------------------------------------------------
		# Columns
		# --------------------------------------------------
		for i in range(rows):
			for j in range(cols):
				h = float(Z[i, j])
				if h <= 0:
					continue

				column = trimesh.creation.box(
					extents=(cell_size, cell_size, h)
				)

				column.apply_translation(
					(
						j * cell_size + cell_size / 2,
						i * cell_size + cell_size / 2,
						base_thickness + h / 2,
					)
				)

				meshes.append(column)

		# --------------------------------------------------
		# Boolean union → single watertight mesh
		# --------------------------------------------------
		solid = trimesh.util.concatenate(meshes)

		return solid


	Z = counts.astype(float)
	Z = normalize_z(Z)
	Z = 2.0 * Z
	order = np.argsort(Z[0, :])

	# reorder columns
	Z = Z[:, order]

	mesh = heightmap_to_column_mesh(
		Z,
		cell_size=1.0,  # 1 cm squares
		base_thickness=1.0  # 1 cm base
	)

	mesh.export("layer_condition_columns.stl")
