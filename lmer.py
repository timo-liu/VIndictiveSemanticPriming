import statsmodels.api as sm
import statsmodels.formula.api as smf
from Definitions.Utils import *
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-i", "--input", type=str, required=False, help="Input csv file",
                           default="data/all naming subjects.xlsx")
    args = argparser.parse_args()

    data = parse_data(args.input,
                      extract_dict={
                          "subject" : "subject",
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

    # RT cleanup
    data["RT"] = pd.to_numeric(data["RT"], errors="coerce")
    data = data.dropna(subset=["RT"])

    data["relation"] = (
        data["first_associate"]
        .combine_first(data["other_associate"])
        .combine_first(data["primecondition"])
    )

    # data = data.groupby(
    #     ["prime", "target", "isi", "relation"],
    #     as_index=False
    # )["RT"].mean()

    data["pair"] = data["prime"].astype(str) + "_" + data["target"].astype(str)

    md = smf.mixedlm("RT ~ relation + isi", data, groups = data["subject"])
    mdf = md.fit(method=["nm"])
    print(mdf.summary())