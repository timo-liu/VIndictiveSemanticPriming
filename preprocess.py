from Definitions.Utils import parse_data
import argparse
import pandas as pd

def add_relation_column(
    dataset,
    relations_path,
    sheet_name,
    prime_col,
    relation_col,
    new_col
):
    """
    Adds a relation-value column to dataset from a given relations sheet.
    """

    # load relation sheet
    relations = parse_data(
        relations_path,
        sheet_name=sheet_name,
        extract_dict={
            "prime": prime_col,
            "target": "TARGET",
            "relation": relation_col
        }
    )

    # normalize strings
    for df in (dataset, relations):
        df["prime"] = df["prime"].astype(str).str.strip()
        df["target"] = df["target"].astype(str).str.strip()

    # build unordered pair keys
    dataset["pair_key"] = dataset.apply(
        lambda r: tuple(sorted((r["prime"], r["target"]))),
        axis=1
    )
    relations["pair_key"] = relations.apply(
        lambda r: tuple(sorted((r["prime"], r["target"]))),
        axis=1
    )

    # rename relation column
    relations = relations.rename(columns={"relation": new_col})

    # merge relation values
    dataset = dataset.merge(
        relations[["pair_key", new_col]],
        on="pair_key",
        how="left"
    )

    return dataset.drop(columns=["pair_key"])



if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-i", "--input", required=False, default="data/all naming subjects.xlsx")
    arg_parser.add_argument("-r", "--relations", required=False, default="data/items_spreadsheet.xls")
    arg_parser.add_argument("-o", "--output")
    args = arg_parser.parse_args()

    # ---------- Load main dataset ----------
    dataset = parse_data(args.input,
                         extract_dict={
                             "subject" : "Subject",
                             "prime": "prime",
                             "target": "target",
                             "primecondition": "primecond",
                             "RT": "target.RT",
                             "accuracy": "target.ACC",
                             "isi": "isi"
                         })

    # ---------- Add columns from multiple sheets ----------
    dataset = add_relation_column(
        dataset,
        args.relations,
        sheet_name="first associate",
        prime_col="prime_first associate",
        relation_col="relation1_f-t",
        new_col="relation1_f-t"
    )

    dataset = add_relation_column(
        dataset,
        args.relations,
        sheet_name="other associate",
        prime_col="other Assoc",
        relation_col="relation1_o-t",
        new_col="relation1_o-t"
    )

    # let's assume that LSA captures the associative features

    dataset = add_relation_column(
        dataset,
        args.relations,
        sheet_name="first associate",
        prime_col="prime_first associate",
        relation_col="LSA_f-t",
        new_col="LSA_f-t"
    )

    # ---------- Save ----------
    dataset.to_excel(args.output, index=False)
