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

	similarity_relations = [
		"synonym",
		"antonym",
		"superordinate",
		"supraordinate",
		"instrument",
		"functional",
		"perceptual",
		"category"
	]


	def assign_relation(row):
		first = str(row["first_associate"]).strip() if not is_blank(row["first_associate"]) else ""
		other = str(row["other_associate"]).strip() if not is_blank(row["other_associate"]) else ""

		# Check for similarity relations in other_associate first
		if other in similarity_relations:
			return other

		# If not, use first_associate if it's a similarity relation
		if first in similarity_relations:
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
			}
		)
	)

	lsa_values = pd.to_numeric(data["LSA_f-t"], errors="coerce").dropna()
	m = lsa_values.median()
	print(f"LSA_f-t median: {m}")

	# Assign "similar" only for similarity relations below median
	data.loc[
		(data["relation"].isin(similarity_relations)) &
		(data["LSA_f-t"].astype(float) < m),
		"relation"
	] = "similar"

	# Assign "associated" only if not already "similar"
	data.loc[
		(data["LSA_f-t"].astype(float) >= m),
		"relation"
	] = "associated"

	# Keep only "similar" and "associated"
	data = data[data["relation"].isin(["similar", "associated"])].copy()

	# num_similar_trials = len(data[data["relation"] == "similar"])
	# num_associated_trials = len(data[data["relation"] == "associated"])
	# print(f"Number of similar trials: {num_similar_trials}")
	# print(f"Number of associated trials: {num_associated_trials}")
	# num_similar_50 = len(
	# 	data[(data["relation"] == "similar") & (data["isi"] == 50)]
	# )
	# num_similar_1050 = len(
	# 	data[(data["relation"] == "similar") & (data["isi"] == 1050)]
	# )
	# print(f"Number of similar 50 trials: {num_similar_50}")
	# print(f"Number of similar 1050 trials: {num_similar_1050}")
	#
	# num_associated_50 = len(data[data["relation"] == "associated"])
	# num_associated_1050 = len(data[data["relation"] == "associated"])
	# print(f"Number of associated 50 trials: {num_associated_50}")
	# print(f"Number of associated 1050 trials: {num_associated_1050}")

	similar_50 = data[(data["relation"] == "similar") & (data["isi"] == 50)]
	similar_1050 = data[(data["relation"] == "similar") & (data["isi"] == 1050)]
	associated_50 = data[(data["relation"] == "associated") & (data["isi"] == 50)]
	associated_1050 = data[(data["relation"] == "associated") & (data["isi"] == 1050)]

	all_bootstraps = []

	for i in tqdm(range(args.bootstrap)):
		s50 = similar_50.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		s1050 = similar_1050.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		a50 = associated_50.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		a1050 = associated_1050.sample(
			n=args.samples,
			replace=True,
			random_state=i
		)

		boot = pd.concat(
			[s50, s1050, a50, a1050],
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