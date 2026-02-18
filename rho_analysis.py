import json
import argparse
import os
from typing import List
from matplotlib import pyplot as plt

plt.rcParams.update({
	"figure.dpi": 300,
	"savefig.dpi": 300,

	"font.size": 13,
	"axes.titlesize": 15,
	"axes.labelsize": 14,
	"xtick.labelsize": 12,
	"ytick.labelsize": 12,
	"legend.fontsize": 12,

	"axes.linewidth": 1.2,
	"lines.linewidth": 1.5,
	"lines.markersize": 3,
})


# ============================================================
# Utils
# ============================================================

def latex_escape(s):
	return str(s).replace("_", "\\_")

def format_rho_latex(entry, bold=False):
	if entry is None:
		return ""
	rho = entry[0]["rho"] * -100
	p = entry[0]["p"]

	if p < 0.01:
		star = "$^{*}$"
	elif p <= 0.05:
		star = "$^{\\dagger}$"
	else:
		star = ""

	val = f"{rho:.3f}"

	if bold:
		val = f"\\textbf{{{val}}}"

	return f"{val}{star}"

import numpy as np

def bootstrap_mean_ci(entries, ci=95):
	"""
	entries: list of dicts with key 'rho'
	returns mean, lower, upper
	"""
	if not entries:
		return None, None, None

	values = np.array([e["rho"] for e in entries])

	mean = np.mean(values)

	alpha = (100 - ci) / 2
	lower = np.percentile(values, alpha)
	upper = np.percentile(values, 100 - alpha)

	return mean, lower, upper

def format_rho(entry):
	if entry is None:
		return ""
	rho = entry[0]["rho"] * -100
	p = entry[0]["p"]
	return f"{rho:.3f}{'*' if p < 0.05 else ''}"

def format_n(entry):
	if entry is None:
		return ""
	return str(entry[0]["n"])

def sort_primeconditions(primeconditions):
	try:
		return sorted(primeconditions, key=lambda x: float(x))
	except (ValueError, TypeError):
		return sorted(primeconditions, key=str)

def component_sort_key(c):
	if c == "word_embeddings":
		return 0
	if c.startswith("encoder_layer_"):
		return int(c.split("_")[-1])
	return 999


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--rhos", default="rhos.json")
	parser.add_argument("--conditions", type=List[str], nargs="+", default=["nam", "ldt"])
	parser.add_argument("--graphs", type=str, default="rho_graphs")
	args = parser.parse_args()

	assert os.path.exists(args.rhos)
	os.makedirs(args.graphs, exist_ok=True)

	ISIS = [50, 1050]

	with open(args.rhos, "r") as f:
		rhos = json.load(f)

	# ------------------------------------------------------------
	# Container per dataset
	# ------------------------------------------------------------
	conditions = {
		cond: {
			"components": [],
			"rho_dict": {},
			"primeconditions": set()
		}
		for cond in args.conditions
	}

	# ------------------------------------------------------------
	# Load JSON (bootstrap-aware)
	# ------------------------------------------------------------
	for entry in rhos.values():

		dataset = entry["dataset"]
		model = entry["model"]
		component = entry["component"]
		relation = entry["relation"]
		isi = entry["isi"]
		bs = entry.get("bootstrap")

		if dataset not in conditions:
			continue

		conditions[dataset]["primeconditions"].add(relation)

		if component not in conditions[dataset]["components"]:
			conditions[dataset]["components"].append(component)

		rho_store = conditions[dataset]["rho_dict"]

		rho_store.setdefault(model, {})
		rho_store[model].setdefault(component, {})
		rho_store[model][component].setdefault(relation, {})
		rho_store[model][component][relation].setdefault(isi, [])

		rho_store[model][component][relation][isi].append({
			"rho": entry["spearman_rho"],
			"p": entry["p_value"],
			"n": entry["n"],
			"bs": bs
		})
	# ============================================================
	# MEAN + CI LINE PLOTS (All models in one figure)
	# ============================================================

	import numpy as np


	def bootstrap_mean_ci(entries, ci=95):
		if not entries:
			return None, None, None

		values = np.array([e["rho"] for e in entries])

		mean = np.mean(values)
		alpha = (100 - ci) / 2
		lower = np.percentile(values, alpha)
		upper = np.percentile(values, 100 - alpha)

		return mean, lower, upper


	for dataset, data in conditions.items():

		rho_dict = data["rho_dict"]
		components = sorted(data["components"], key=component_sort_key)
		models = sorted(rho_dict.keys())
		primeconditions = sort_primeconditions(data["primeconditions"])

		for relation in primeconditions:
			for isi in ISIS:

				fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=False)

				x = np.arange(len(components))

				plotted_any = False

				for model in models:

					means = []
					lower_err = []
					upper_err = []

					for comp in components:

						entries = (
							rho_dict
							.get(model, {})
							.get(comp, {})
							.get(relation, {})
							.get(isi, [])
						)

						if not entries:
							means.append(np.nan)
							lower_err.append(np.nan)
							upper_err.append(np.nan)
							continue

						mean, lower, upper = bootstrap_mean_ci(entries)

						means.append(mean)
						lower_err.append(mean - lower)
						upper_err.append(upper - mean)

					if not any(np.isfinite(means)):
						continue

					plotted_any = True

					ax.errorbar(
						x,
						means,
						yerr=[lower_err, upper_err],
						fmt='o-',
						capsize=4,
						label=model
					)

				if not plotted_any:
					plt.close(fig)
					continue

				ax.axhline(0, linestyle="--", linewidth=1)

				ax.set_xticks(x)
				ax.set_xticklabels(components, rotation=45, ha="right")

				ax.set_ylabel("Mean Spearman $\\rho$")
				ax.set_xlabel("Component / encoder layer")
				ax.set_title(
					f"{dataset.upper()} — {relation}, ISI {isi} ms"
				)

				ax.legend(
					loc="center left",
					bbox_to_anchor=(1.02, 0.5),
					frameon=False
				)

				fname = f"{dataset}_rel{relation}_isi{isi}_meanCI_lines.png"

				fig.tight_layout(rect=[0, 0, 0.82, 1])

				fig.savefig(
					os.path.join(args.graphs, fname),
					bbox_inches="tight"
				)

				plt.close(fig)
				print(f"Saved {fname}")

	# ============================================================
	# Tables + LaTeX
	# ============================================================

	print("\n" + "=" * 80)
	print("MODEL × COMPONENT TABLES (ρ, * = p < .05)")
	print("=" * 80)

	latex_dir = os.path.join(args.graphs, "latex")
	os.makedirs(latex_dir, exist_ok=True)

	for dataset, data in conditions.items():
		rho_dict = data["rho_dict"]
		components = sorted(data["components"], key=component_sort_key)
		models = sorted(rho_dict.keys())
		primeconditions = sort_primeconditions(data["primeconditions"])

		for relation in primeconditions:
			for isi in ISIS:

				# ---------------- Console table ----------------
				print(f"\nDataset: {dataset.upper()} | Relation: {relation} | ISI: {isi}")
				print("-" * 80)

				print("{:<25}".format("Model"), end="")
				for comp in components:
					print("{:>15}".format(comp), end="")
				print()

				print("-" * (25 + 15 * len(components)))

				for model in models:
					print("{:<25}".format(model), end="")
					for comp in components:
						entry = (
							rho_dict
							.get(model, {})
							.get(comp, {})
							.get(relation, {})
							.get(isi)
						)
						print("{:>15}".format(format_rho(entry)), end="")
					print()

				# ---------------- LaTeX rho table ----------------
				rho_path = os.path.join(
					latex_dir,
					f"{dataset}_rel{relation}_isi{isi}.tex"
				)

				# --- Precompute column maxima (after * -100) ---
				column_max = {}

				for model in models:
					values = []
					for comp in components:
						entry = rho_dict.get(model, {}).get(comp, {}).get(relation, {}).get(isi)
						if entry is not None:
							values.append(entry[0]["rho"] * -100)

					column_max[model] = max(values) if values else None

				with open(rho_path, "w") as f:
					f.write("\\begin{table*}[ht]\n\\centering\n\\small\n")
					colspec = "l" + "c" * len(models)
					f.write("\\begin{adjustbox}{max width=\\textwidth}\n")
					f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
					f.write("\\toprule\n")
					f.write("Component & " + " & ".join(latex_escape(m) for m in models) + " \\\\\n")
					f.write("\\midrule\n")

					for comp in components:
						row = [latex_escape(comp)]
						for model in models:
							entry = rho_dict.get(model, {}).get(comp, {}).get(relation, {}).get(isi)

							is_bold = False
							if entry is not None and column_max[model] is not None:
								value = entry[0]["rho"] * -100
								if value == column_max[model]:
									is_bold = True

							row.append(format_rho_latex(entry, bold=is_bold))

						f.write(" & ".join(row) + " \\\\\n")

					f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n")
					f.write(
						f"\\caption{{Spearman $\\rho$ between cosine similarity and RT "
						f"({dataset.upper()}, relation {relation}, ISI={isi} ms). "
						f"$^*$ $p<.01$, $^\\dagger$ $.01\\leq p \\leq .05$. "
						f"Bold = largest value within model.}}\n"
					)
					f.write(f"\\label{{tab:{dataset}_rel{relation}_isi{isi}}}\n")
					f.write("\\end{table*}\n")

				# ---------------- LaTeX n table ----------------
				n_path = os.path.join(
					latex_dir,
					f"{dataset}_rel{relation}_isi{isi}_n.tex"
				)

				with open(n_path, "w") as f:
					f.write("\\begin{table}[ht]\n\\centering\n\\small\n")
					colspec = "l" + "c" * len(models)
					f.write("\\begin{adjustbox}{max width=\\textwidth}\n")
					f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
					f.write("\\toprule\n")
					f.write("Component & " + " & ".join(latex_escape(m) for m in models) + " \\\\\n")
					f.write("\\midrule\n")

					for comp in components:
						row = [latex_escape(comp)]
						for model in models:
							entry = rho_dict.get(model, {}).get(comp, {}).get(relation, {}).get(isi)
							row.append(format_n(entry))
						f.write(" & ".join(row) + " \\\\\n")

					f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n")
					f.write(
						f"\\caption{{Number of observations ($n$) used for Spearman $\\rho$ "
						f"({dataset.upper()}, relation {relation}, ISI={isi} ms).}}\n"
					)
					f.write(f"\\label{{tab:{dataset}_rel{relation}_isi{isi}_n}}\n")
					f.write("\\end{table}\n")

				print(f"Wrote LaTeX tables: {rho_path}, {n_path}")
