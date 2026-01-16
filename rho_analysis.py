import json
import argparse
import os
from typing import List
from matplotlib import pyplot as plt

# region Utils
def latex_escape(s):
	return s.replace("_", "\\_")

def format_rho_latex(entry):
	if entry is None:
		return ""
	rho = entry["rho"]
	p = entry["p"]
	star = "$^{*}$" if p < 0.01 else "$^{\\dagger}$" if 0.01 <= p <= 0.05 else ""
	return f"{rho:.3f}{star}"


def format_rho(entry):
	if entry is None:
		return ""
	rho = entry["rho"]
	p = entry["p"]
	return f"{rho:.3f}{'*' if p < 0.05 else ''}"

# endregion Utils

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("--rhos", default="rhos.json")
	parser.add_argument("--conditions", type = List[str], nargs='+', default=["nam", "ldt"])
	parser.add_argument("--graphs", type = str, default="rho_graphs")
	args = parser.parse_args()

	assert os.path.exists(args.rhos)
	os.makedirs(args.graphs, exist_ok=True)

	PRIMECONDITIONS = [1, 2, 3, 4]
	ISIS = [50, 1050]

	with open(args.rhos, "r") as f:
		rhos = json.load(f)

	conditions = {
		condition : {
			"components" : [],
			"rho_dict" : {}
		}
		for condition in args.conditions
	}

	for entry in rhos.values():
		dataset = entry["dataset"]
		model = entry["model"]
		component = entry["component"]
		primecondition = entry["primecondition"]
		isi = entry["isi"]

		# Track components seen per condition
		if component not in conditions[dataset]["components"]:
			conditions[dataset]["components"].append(component)

		# Initialize nested dicts
		conditions[dataset]["rho_dict"].setdefault(model, {})
		conditions[dataset]["rho_dict"][model].setdefault(component, {})
		conditions[dataset]["rho_dict"][model][component].setdefault(primecondition, {})

		# Store by ISI
		conditions[dataset]["rho_dict"][model][component][primecondition][isi] = {
			"rho": entry["spearman_rho"],
			"p": entry["p_value"],
			"n": entry["n"]
		}


	def component_sort_key(c):
		if c == "word_embeddings":
			return 0
		if c.startswith("encoder_layer_"):
			return int(c.split("_")[-1])
		return 999


	for dataset, data in conditions.items():
		rho_dict = data["rho_dict"]
		components = sorted(data["components"], key=component_sort_key)

		for primecondition in PRIMECONDITIONS:
			for isi in ISIS:

				plt.figure(figsize=(10, 6))
				plotted_any = False

				for model, model_data in rho_dict.items():
					xs, ys = [], []

					for comp in components:
						entry = (
							model_data
							.get(comp, {})
							.get(primecondition, {})
							.get(isi)
						)
						if entry is not None:
							xs.append(comp)
							ys.append(entry["rho"])

					# DistilBERT stops early automatically
					if ys:
						plt.plot(xs, ys, marker="o", label=model)
						plotted_any = True

						# Add significance markers
						for x, y, comp in zip(xs, ys, xs):
							p = (
								model_data
								.get(comp, {})
								.get(primecondition, {})
								.get(isi, {})
								.get("p")
							)
							if p is not None and p < 0.05:
								plt.text(
									x,
									y,
									"*",
									color="red",
									fontsize=14,
									ha="center",
									va="bottom"
								)

				if not plotted_any:
					plt.close()
					continue

				plt.axhline(0, linestyle="--", linewidth=1)
				plt.xticks(rotation=45, ha="right")
				plt.ylabel("Spearman ρ (cosine vs RT)")
				plt.xlabel("Component / Encoder layer")
				plt.title(
					f"{dataset.upper()} | primecondition={primecondition} | isi={isi}"
				)
				plt.legend()
				plt.tight_layout()

				fname = f"{dataset}_pc{primecondition}_isi{isi}_model_comparison.png"
				plt.savefig(os.path.join(args.graphs, fname))
				plt.close()

				print(f"Saved {fname}")
print("\n" + "=" * 80)
print("MODEL × COMPONENT TABLES (ρ, * = p < .05)")
print("=" * 80)

for dataset, data in conditions.items():
	rho_dict = data["rho_dict"]
	components = sorted(data["components"], key=component_sort_key)
	models = sorted(rho_dict.keys())

	for primecondition in PRIMECONDITIONS:
		for isi in ISIS:

			print(f"\nDataset: {dataset.upper()} | "
				  f"PrimeCondition: {primecondition} | ISI: {isi}")
			print("-" * 80)

			# Header
			header = ["Model"] + components
			print("{:<25}".format(header[0]), end="")
			for comp in components:
				print("{:>15}".format(comp), end="")
			print()

			print("-" * (25 + 15 * len(components)))

			# Rows
			for model in models:
				print("{:<25}".format(model), end="")

				for comp in components:
					entry = (
						rho_dict
						.get(model, {})
						.get(comp, {})
						.get(primecondition, {})
						.get(isi)
					)
					print("{:>15}".format(format_rho(entry)), end="")

				print()

				latex_dir = os.path.join(args.graphs, "latex")
				os.makedirs(latex_dir, exist_ok=True)

				for dataset, data in conditions.items():
					rho_dict = data["rho_dict"]
					components = sorted(data["components"], key=component_sort_key)
					models = sorted(rho_dict.keys())

					for primecondition in PRIMECONDITIONS:
						for isi in ISIS:

							fname = f"{dataset}_pc{primecondition}_isi{isi}.tex"
							path = os.path.join(latex_dir, fname)

							with open(path, "w") as f:
								# Table header
								f.write("\\begin{table}[ht]\n")
								f.write("\\centering\n")
								f.write("\\small\n")

								colspec = "l" + "c" * len(models)
								f.write("\\begin{adjustbox}{max width=\\textwidth}\n")
								f.write(f"\\begin{{tabular}}{{{colspec}}}\n")
								f.write("\\toprule\n")

								header = ["Component"] + [latex_escape(m) for m in models]
								f.write(" & ".join(header) + " \\\\\n")
								f.write("\\midrule\n")

								# Rows
								for comp in components:
									row = [latex_escape(comp)]
									for model in models:
										entry = rho_dict.get(model, {}).get(comp, {}).get(primecondition, {}).get(isi)
										row.append(format_rho_latex(entry))
									f.write(" & ".join(row) + " \\\\\n")

								f.write("\\bottomrule\n")
								f.write("\\end{tabular}\n")
								f.write("\\end{adjustbox}\n")
								caption = (
									f"Spearman $\\rho$ between cosine similarity and RT "
									f"({dataset.upper()}, prime condition {primecondition}, ISI={isi} ms). "
									f"$^*$ indicates $p < .05$."
								)
								f.write(f"\\caption{{{caption}}}\n")
								f.write(f"\\label{{tab:{dataset}_pc{primecondition}_isi{isi}}}\n")
								f.write("\\end{table}\n")

							print(f"Wrote LaTeX table: {path}")

