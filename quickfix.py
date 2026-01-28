from Definitions.Utils import *

saver = "data/ldt.xlsx"
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
    return str(row["primecondition"])

data["relation"] = data.apply(assign_relation, axis=1)

data["relation"] = (
    data["relation"]
    .astype(str)
    .str.strip()
    .replace(
        {
            "unclassifed": "unclassified",
            "unclassified": "unclassified",
            "unclassified ": "unclassified",  # keeps canonical form
        }
    )
)

data = data.groupby(
    ["prime", "target", "isi", "relation"],
    as_index=False
)["RT"].mean()

compressed = "data/ldt_compressed.xlsx"
data.to_excel(compressed, index=False)