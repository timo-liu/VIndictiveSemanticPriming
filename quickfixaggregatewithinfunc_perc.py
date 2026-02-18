from Definitions.Utils import *
from tqdm import tqdm

"""
Sort relations into featural buckets
"""

import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i","--input",help="input excel file")
	parser.add_argument("-o","--output",help="output excel file")
	parser.add_argument("--bootstrap", type=int, default=500)
	parser.add_argument("--samples", type=int, default=1000)
	args = parser.parse_args()

	saver = args.input
	compressed = args.output

	relatedness_dir = {
		1.0 : "first-associate-related",
		2.0 : "first-associate-unrelated",
		3.0 : "other_associate-related",
		4.0 : "other-associate-unrelated",
	}

	data = parse_data(
			saver,
			extract_dict={
				"prime": "prime",
				"target": "target",
				"primecondition": "primecondition",
				"RT": "RT",
				"accuracy": "accuracy",
				"isi": "isi",
				"first_associate": "relation1_f-t",
				"other_associate": "relation1_o-t",
				"LSA_f-t" : "LSA_f-t"
			}
		)
	data = data[data["accuracy"] == 1]
	data["RT"] = pd.to_numeric(data["RT"], errors="coerce")
	data = data.dropna(subset=["RT"])

	def assign_relation(row):
		first = str(row["first_associate"]).strip() if not is_blank(row["first_associate"]) else ""
		other = str(row["other_associate"]).strip() if not is_blank(row["other_associate"]) else ""

		interest_list = ["perceptual", "functional"]

		# Check for similarity relations in other_associate first
		if other in interest_list:
			return other

		# If not, use first_associate if it's a similarity relation
		if first in interest_list:
			return first

		# Otherwise fallback to first_associate or other_associate if present
		if first:
			return first
		if other:
			return other

		# Final fallback based on primecondition
		key = float(row["primecondition"])
		return relatedness_dir.get(key, "unknown")


	# Apply
	data["relation"] = data.apply(assign_relation, axis=1)

	# Clean up spelling / casing
	data["relation"] = (
		data["relation"]
		.astype(str)
		.str.strip()
		.replace(
			{
				"unclassfied": "unclassified",
				"unclassifed": "unclassified",
				"unclassified ": "unclassified",
				"antonymn": "antonym",
				"Instrument": "instrument",
				"functional property": "functional",
				"perceptual property": "perceptual"
			}
		)
	)

	functional_50 = data[(data["relation"] == "functional") & (data["isi"] == 50)]
	functional_1050 = data[(data["relation"] == "functional") & (data["isi"] == 1050)]

	perceptual_50 = data[(data["relation"] == "perceptual") & (data["isi"] == 50)]
	perceptual_1050 = data[(data["relation"] == "perceptual") & (data["isi"] == 1050)]

	print(f"Functional, ISI=50:   {len(functional_50)} rows")
	print(f"Functional, ISI=1050: {len(functional_1050)} rows")
	print(f"Perceptual, ISI=50:   {len(perceptual_50)} rows")
	print(f"Perceptual, ISI=1050: {len(perceptual_1050)} rows")

	all_bootstraps = []

	for i in tqdm(range(args.bootstrap)):
		f50 = functional_50.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		f1050 = functional_1050.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		p50 = perceptual_50.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		p1050 = perceptual_1050.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		boot = pd.concat(
			[f50, f1050, p50, p1050],
			ignore_index=True
		)

		# Collapse by item
		boot = (
			boot
			.groupby(
				["prime", "target", "isi", "relation"],
				as_index=False
			)["RT"]
			.mean()
		)

		# Add bootstrap ID
		boot["bs"] = i

		all_bootstraps.append(boot)

	final_df = pd.concat(all_bootstraps, ignore_index=True)

	final_df.to_excel(args.output, index=False)

	print(f"\nSaved bootstrapped dataset with {args.bootstrap} samples to:")
	print(args.output)
