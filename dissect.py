import pickle
import argparse
import numpy as np
import scipy.stats as st
from matplotlib import pyplot as plt
from collections import defaultdict, Counter
import os

top_k = 5

def layer_sort_key(layer):
    if layer == "word_embeddings":
        return -1
    if layer.startswith("encoder_layer_"):
        return int(layer.split("_")[-1])
    return float("inf")  # anything unexpected goes to the end


if __name__ == "__main__":
    # region pickle
    def leaf():
        return {"rhos": [], "ps": []}


    def level3():
        return defaultdict(leaf)


    def level2():
        return defaultdict(level3)


    def level1():
        return defaultdict(level2)
    # endregion pickle

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "-i", "--input", required=False, help="Input file path", default="rsa_acc_whole.pkl"
    )
    arg_parser.add_argument(
        "-o", "--output", required=False, help="Graphs go brrr", default="layer_rankings"
    )

    args = arg_parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    with open(args.input, "rb") as f:
        rsa_acc_whole = pickle.load(f)

    top3_layers = defaultdict(lambda: defaultdict(dict))
    all_layers = set()

    for task, condition_data in rsa_acc_whole.items():
        for condition, layer_data in condition_data.items():
            per_model = defaultdict(list)
            for layer, model_data in layer_data.items():
                all_layers.add(layer)
                for model, rho_data in model_data.items():
                    if "distil" in model or "albert-base" in model:
                        continue
                    rizz = [1 - r for r in rho_data["rhos"]]
                    rho_mean = np.mean(rizz)
                    error = st.sem(rizz)

                    per_model[model].append((layer, rho_mean, error))

            for model, layers in per_model.items():
                layers = sorted(layers, key=lambda x: x[1])  # lower is better
                top3_layers[condition][task][model] = {layer for layer, _, _ in layers[:top_k]}

    condition_task_layer_counts = defaultdict(lambda: defaultdict(Counter))

    for condition, task_data in top3_layers.items():
        for task, model_data in task_data.items():
            for layerset in model_data.values():
                condition_task_layer_counts[condition][task].update(layerset)

    condition_layer_counts = defaultdict(Counter)

    for condition, task_data in top3_layers.items():

        model_union = defaultdict(set)

        for task, model_data in task_data.items():
            for model, layerset in model_data.items():
                model_union[model].update(layerset)

        for layerset in model_union.values():
            condition_layer_counts[condition].update(layerset)

    # 4. Plot aggregated per condition
    for condition, layer_counter in condition_layer_counts.items():
        layers = sorted(all_layers, key=layer_sort_key)
        counts = [layer_counter.get(l, 0) for l in layers]

        plt.figure()
        plt.bar(layers, counts)
        plt.xlabel("Layer")
        plt.ylabel("Number of models")
        plt.title(f"Top-{top_k} Layer Frequency — {condition}")
        plt.xticks(rotation=45, ha="right", fontsize=8)
        plt.tight_layout()

        save_path = os.path.join(args.output, f"top{top_k}_layer_frequency_{condition}.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

    for condition, task_data in condition_task_layer_counts.items():
        for task, layer_counter in task_data.items():
            layers = sorted(all_layers, key=layer_sort_key)
            counts = [layer_counter.get(l, 0) for l in layers]

            plt.figure()
            plt.bar(layers, counts)
            plt.xlabel("Layer")
            plt.ylabel("Number of models")
            plt.title(f"{task} — {condition}")
            plt.xticks(rotation=45, ha="right", fontsize=8)
            plt.tight_layout()

            save_path = os.path.join(args.output, f"top{top_k}_layer_frequency_{task}_{condition}.png")
            plt.savefig(save_path, dpi=300)
            plt.close()
