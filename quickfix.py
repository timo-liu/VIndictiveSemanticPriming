from Definitions.Utils import *
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument("-i","--input",help="input excel file")
	parser.add_argument("-o","--output",help="output excel file")
	args = parser.parse_args()

	relatedness_dir = {
		1.0 : "first-associate-related",
		2.0 : "first-associate-unrelated",
		3.0 : "other_associate-related",
		4.0 : "other-associate-related",
	}

	saver = args.input
	compressed = args.output

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
			}
		)
	data = data[data["accuracy"] == 1]
	data["RT"] = pd.to_numeric(data["RT"], errors="coerce")
	data = data.dropna(subset=["RT"])

	def assign_relation(row):
		if not is_blank(row["first_associate"]):
			return str(row["first_associate"])
		if not is_blank(row["other_associate"]):
			return str(row["other_associate"])
		key = float(row["primecondition"])
		return relatedness_dir.get(key, "unknown")

	data["relation"] = data.apply(assign_relation, axis=1)

	data["relation"] = (
		data["relation"]
		.astype(str)
		.str.strip()
		.replace(
			{
				"unclassfied": "unclassified",
				"unclassifed": "unclassified",
				"unclassified": "unclassified",
				"unclassified ": "unclassified",
				"antonymn" : "antonym",
				"Instrument" : "instrument",
			}
		)
	)

	data = data.groupby(
		["prime", "target", "isi", "relation"],
		as_index=False
	)["RT"].mean()

	counts = data["relation"].value_counts()

	for relation, n in counts.items():
		if n < 200:
			print(f"Dropping relation {relation} ({n} rows)")
		else:
			print(f"Not dropping relation {relation} ({n} rows)")

	data = data[data["relation"].map(counts) >= 200]

	data.to_excel(compressed, index=False)