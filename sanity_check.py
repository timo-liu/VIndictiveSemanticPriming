import scipy.stats as stats
import argparse
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--input",help="input excel file")
    args = parser.parse_args()

    data = pd.read_excel(args.input)

    col = "RT"

    # --------------------------------------------------
    # 1) Within ISI: similar vs associated
    # --------------------------------------------------
    isis = data["isi"].unique()

    for isi in isis:
        similar_data = data.loc[
            (data["relation"] == "similar") & (data["isi"] == isi)
        ].copy()

        associated_data = data.loc[
            (data["relation"] == "associated") & (data["isi"] == isi)
        ].copy()

        x = similar_data[col].astype(float).dropna()
        y = associated_data[col].astype(float).dropna()

        if len(x) > 1 and len(y) > 1:
            t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)
            print(f"\nTwo-sided Welch t-test (similar vs associated | isi={isi})")
            print(f"t = {t_stat:.6f}")
            print(f"p = {p_value:.6f}")

    # --------------------------------------------------
    # 2) Between ISIs within each condition
    # --------------------------------------------------
    relations = data["relation"].unique()

    for relation in relations:
        condition_data = data.loc[data["relation"] == relation]

        isis = condition_data["isi"].unique()

        # pairwise ISI comparisons
        for i in range(len(isis)):
            for j in range(i+1, len(isis)):
                isi1 = isis[i]
                isi2 = isis[j]

                x = condition_data.loc[
                    condition_data["isi"] == isi1, col
                ].astype(float).dropna()

                y = condition_data.loc[
                    condition_data["isi"] == isi2, col
                ].astype(float).dropna()

                if len(x) > 1 and len(y) > 1:
                    t_stat, p_value = stats.ttest_ind(x, y, equal_var=False)

                    print(f"\nTwo-sided Welch t-test ({relation} | isi={isi1} vs isi={isi2})")
                    print(f"t = {t_stat:.6f}")
                    print(f"p = {p_value:.6f}")