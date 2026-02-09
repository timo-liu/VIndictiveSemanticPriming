import csv
import argparse
from matplotlib import pyplot as plt
from collections import defaultdict
import numpy as np
import scipy.stats as st
import os
import pickle

if __name__ == '__main__':
    def leaf():
        return {"rhos": [], "ps": []}


    def level3():
        return defaultdict(leaf)


    def level2():
        return defaultdict(level3)


    def level1():
        return defaultdict(level2)
    # endregion pickle
    """
    Open up the rsv and compute means
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--data", help="input csv file", required=False, default="rsa_meta.csv")
    argparser.add_argument("--graphs", help="graph dir", required=False, default="rsa_graphs")
    argparser.add_argument("--pickle", help="pickle to output", action="store_true")
    args = argparser.parse_args()

    os.makedirs(args.graphs, exist_ok=True)

    # columns = task | condition | component | model | rt_boot | emb_boot | rho | p

    adder = defaultdict(level1)
    # task -> condition -> model -> component

    with open(args.data) as rsa:
        reader = csv.DictReader(rsa)
        for row in reader:
            task = row["task"]
            condition = row["condition"]
            component = row["component"]
            model = row["model"]
            rt_boot = row["rt_boot"]
            emb_boot = row["emb_boot"]
            rho = row["rho"]
            adder[task][condition][model][component]["rhos"].append(float(rho))

    if args.pickle:
        with open("rsa_rho_dump.pkl", "wb") as w:
            pickle.dump(adder, w)

    for task, condition_data in adder.items():
        for condition, model_data in condition_data.items():
            for model, component_data in model_data.items():
                ys = [1 - np.mean(rho["rhos"]) for rho in component_data.values()]
                serrs = [st.sem([1 - r for r in rho["rhos"]]) for rho in component_data.values()]
                # try doing analysis with 1-r as std dev
                labels = list(component_data.keys())
                ys, yerrs, labels = zip(*sorted(zip(ys, serrs, labels), key = lambda x: x[0], reverse=False))
                plt.bar(labels, ys, yerr=yerrs)
                plt.xticks(rotation=45, fontsize=6)
                plt.xlabel("component")
                plt.ylabel("rho")
                plt.title(f"Best to Worst Component RSA correlations for task {task}, model {model}, condition {condition}", fontsize=6)
                plt.tight_layout()
                graph_name = f"{task}_{condition}_{model}.png"
                graph_path = os.path.join(args.graphs, graph_name)
                plt.savefig(graph_path)
                plt.clf()

