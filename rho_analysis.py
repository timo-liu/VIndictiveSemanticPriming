import json
import argparse
import os
from typing import List
from matplotlib import pyplot as plt

# ============================================================
# Utils
# ============================================================

def latex_escape(s):
	return str(s).replace("_", "\\_")

def format_rho_latex(entry):
	if entry is None:
		return ""
	rho = entry["rho"]
	p = entry["p"]
	if p < 0.01:
		star = "$^{*}$"
	elif p <= 0.05:
		star = "$^{\\dagger}$"
	else:
		star = ""
	return f"{rho:.3f}{star}"

def format_rho(entry):
	if entry is None:
		return ""
	rho = entry["rho"]
	p = entry["p"]
	return f"{rho:.3f}{'*' if p < 0.05 else ''}"

def format_n(entry):
	if entry is None:
		return ""
	return str(entry["n"])

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
	# Load JSON
	# ------------------------------------------------------------
	for entry in rhos.values():
		dataset = entry["dataset"]
		model = entry["model"]
		component = entry["component"]
		relation = entry["relation"]
		isi = entry["isi"]

		if dataset not in conditions:
			continue

		conditions[dataset]["primeconditions"].add(relation)

		if component not in conditions[dataset]["components"]:
			conditions[dataset]["components"].append(component)

		conditions[dataset]["rho_dict"].setdefault(model, {})
		conditions[dataset]["rho_dict"][model].setdefault(component, {})
		conditions[dataset]["rho_dict"][model][component].setdefault(relation, {})

		conditions[dataset]["rho_dict"][model][component][relation][isi] = {
			"rho": entry["spearman_rho"],
			"p": entry["p_value"],
			"n": entry["n"]
		}

	# ============================================================
	# Plotting
	# ============================================================

	for dataset, data in conditions.items():
		rho_dict = data["rho_dict"]
		components = sorted(data["components"], key=component_sort_key)
		primeconditions = sort_primeconditions(data["primeconditions"])

		for relation in primeconditions:
			for isi in ISIS:

				plt.figure(figsize=(10, 6))
				plotted_any = False

				for model, model_data in rho_dict.items():
					xs, ys = [], []

					for comp in components:
						entry = (
							model_data
							.get(comp, {})
							.get(relation, {})
							.get(isi)
						)
						if entry is not None:
							xs.append(comp)
							ys.append(entry["rho"])

					if ys:
						plt.plot(xs, ys, marker="o", label=model)
						plotted_any = True

						for x, y, comp in zip(xs, ys, xs):
							p = (
								model_data
								.get(comp, {})
								.get(relation, {})
								.get(isi, {})
								.get("p")
							)
							if p is not None and p < 0.05:
								plt.text(x, y, "*", color="red",
										 fontsize=14, ha="center", va="bottom")

				if not plotted_any:
					plt.close()
					continue

				plt.axhline(0, linestyle="--", linewidth=1)
				plt.xticks(rotation=45, ha="right")
				plt.ylabel("Spearman ρ (cosine vs RT)")
				plt.xlabel("Component / Encoder layer")
				plt.title(f"{dataset.upper()} | relation={relation} | isi={isi}")
				plt.legend()
				plt.tight_layout()

				fname = f"{dataset}_rel{relation}_isi{isi}_model_comparison.png"
				plt.savefig(os.path.join(args.graphs, fname))
				plt.close()

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

				with open(rho_path, "w") as f:
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
							row.append(format_rho_latex(entry))
						f.write(" & ".join(row) + " \\\\\n")

					f.write("\\bottomrule\n\\end{tabular}\n\\end{adjustbox}\n")
					f.write(
						f"\\caption{{Spearman $\\rho$ between cosine similarity and RT "
						f"({dataset.upper()}, relation {relation}, ISI={isi} ms). "
						f"$^*$ $p<.01$, $^\\dagger$ $.01\\leq p \\leq .05$.}}\n"
					)
					f.write(f"\\label{{tab:{dataset}_rel{relation}_isi{isi}}}\n")
					f.write("\\end{table}\n")

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
