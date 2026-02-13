import os
import argparse
import numpy as np
from collections import defaultdict
from scipy.stats import spearmanr
from statistics import mean
from tqdm import tqdm
import csv


TARGET_MODEL = ""
TARGET_COMPONENT_PREFIX = ""


# -------------------------------------------------
# Sorting / printing utilities
# -------------------------------------------------

def component_sort_key(c):
	if c == "word_embeddings":
		return (-1, 0)
	if c.startswith("encoder_layer_"):
		return (0, int(c.split("_")[-1]))
	return (1, c)


def pretty_print_table(rsa_dict, precision=3):
	for task, task_data in rsa_dict.items():
		print(f"\n====================")
		print(f"Task: {task.upper()}")
		print(f"====================")

		components = set()
		models = set()

		for condition_data in task_data.values():
			for component, model_data in condition_data.items():
				components.add(component)
				models.update(model_data.keys())

		components = sorted(components, key=component_sort_key)
		models = sorted(models)

		for condition in sorted(task_data.keys()):
			print(f"\n--- Condition: {condition} ---")

			header = ["Component"] + models
			col_width = max(14, max(len(h) for h in header) + 2)
			print("".join(h.ljust(col_width) for h in header))
			print("-" * col_width * len(header))

			for component in components:
				row = [component.ljust(col_width)]

				for model in models:
					if (
						component in task_data[condition]
						and model in task_data[condition][component]
					):
						entry = task_data[condition][component][model]
						rho = entry["rho"]
						p = entry["p"]

						if p < 0.01:
							sig = "*"
						elif p < 0.05:
							sig = "†"
						else:
							sig = ""

						value = f"{rho:.{precision}f}{sig}"
					else:
						value = "—"

					row.append(value.ljust(col_width))

				print("".join(row))

		print("\n* p < 0.01,  † 0.01 ≤ p < 0.05")


# -------------------------------------------------
# Filename parsing
# -------------------------------------------------

def process_name(file_name: str):
	prefix, component = file_name.rsplit("__", 1)
	component = component.replace(".npz", "").rsplit("_", 1)[0]
	task, model = prefix.split("_", 1)
	return task, model, component


# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--dir", type=str, default="D:/heatmaps")
	parser.add_argument("--output", type=str, default="rsa_meta.csv")
	args = parser.parse_args()

	meta_path = os.path.join(args.dir, args.output)

	meta_exists = os.path.exists(meta_path)
	meta_file = open(meta_path, "a", newline="")
	meta_writer = csv.DictWriter(
		meta_file,
		fieldnames=[
			"task",
			"condition",
			"component",
			"model",
			"rt_boot",
			"emb_boot",
			"rho",
			"p",
		]
	)

	if not meta_exists:
		meta_writer.writeheader()

	# ---------------------------------------------
	# Match EMB / RT files
	# ---------------------------------------------

	matcher = defaultdict(dict)

	for file in tqdm(os.listdir(args.dir), desc="Scanning heatmap files"):
		if not file.endswith(".npz") or not ("ldt" in file or "nam" in file ):
			continue
		if TARGET_MODEL not in file:
			continue
		prefix, suffix = file.rsplit("_", 1)
		match_type = suffix.replace(".npz", "")
		matcher[prefix][match_type] = file

	# ---------------------------------------------
	# Streaming RSA accumulator
	# ---------------------------------------------

	# task -> condition -> component -> model -> stats
	rsa_acc = defaultdict(
		lambda: defaultdict(
			lambda: defaultdict(
				lambda: defaultdict(lambda: {"rhos": [], "ps": []})
			)
		)
	)

	# ---------------------------------------------
	# Stream EMB + RT → RSA
	# ---------------------------------------------

	flush_every = 5_000
	i = 0

	for prefix, files in tqdm(matcher.items(), desc="Streaming EMB/RT + RSA"):
		if "EMB" not in files or "RT" not in files:
			continue

		emb_npz = np.load(os.path.join(args.dir, files["EMB"]))
		rt_npz = np.load(os.path.join(args.dir, files["RT"]))

		task, model, component = process_name(files["EMB"])

		# Pre-index EMB keys by condition
		emb_by_condition = defaultdict(dict)
		for emb_key in emb_npz.files:
			if "mean" in emb_key:
				continue
			cond, boot = emb_key.rsplit("__", 1)
			cond = cond.replace("__", "_")
			emb_by_condition[cond][boot] = emb_key

		# tqdm over RT conditions
		for rt_key in tqdm(
				rt_npz.files,
				desc=f"RSA ({task} | {component})",
				leave=False
		):
			if "mean" in rt_key:
				continue
			condition, boot = rt_key.rsplit("__", 1)
			condition = condition.replace("__", "_")

			if condition not in emb_by_condition:
				continue

			rt = rt_npz[rt_key]
			rt_flat = rt[np.triu_indices_from(rt, k=1)]

			# tqdm over boots for this condition
			ek = emb_by_condition[condition][boot]
			i += 1
			emb = emb_npz[ek]
			emb_flat = emb[np.triu_indices_from(emb, k=1)]

			rho, p = spearmanr(rt_flat, emb_flat)

			# accumulate in-memory stats (for current run summary)
			acc = rsa_acc[task][condition][component][model]
			acc["rhos"].append(float(rho))
			acc["ps"].append(float(p))

			# stream to meta file
			meta_writer.writerow({
				"task": task,
				"condition": condition,
				"component": component,
				"model": model,
				"rt_boot": boot,
				"emb_boot": ek.split("__")[-1],
				"rho": float(rho),
				"p": float(p),
			})

			if i % flush_every == 0:
				meta_file.flush()
				i = 0

		emb_npz.close()
		rt_npz.close()

	# ---------------------------------------------
	# Reduce stats
	# ---------------------------------------------

	rsa_dict = defaultdict(
		lambda: defaultdict(lambda: defaultdict(dict))
	)

	for task, task_data in rsa_acc.items():
		for condition, cond_data in task_data.items():
			for component, comp_data in cond_data.items():
				for model, stats in comp_data.items():
					rsa_dict[task][condition][component][model] = {
						"rho": float(np.mean(stats["rhos"])),
						"stds": float(np.std(stats["rhos"])),
						"p": float(np.min(stats["ps"])),
					}

	pretty_print_table(rsa_dict)
