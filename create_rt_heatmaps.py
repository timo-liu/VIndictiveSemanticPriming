from Definitions.Utils import *
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import pickle

"""
RT dissimilarity heatmaps with cached metadata and trial cache
"""


def plot_heatmaps(heat_maps, save_dir=None, figsize=(6, 5), cmap="viridis"):
    for key, matrix in heat_maps.items():
        plt.figure(figsize=figsize)
        matrix_np = np.asarray(matrix)

        ax = sns.heatmap(
            matrix_np,
            cmap=cmap,
            square=True,
            annot=False,
            cbar=True
        )

        ax.set_xticks([])
        ax.set_yticks([])
        plt.title(f"Heatmap: {key}")
        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(
                save_dir,
                f"rt_heatmap_{'__'.join(map(str, key))}.png"
            )
            plt.savefig(save_path, dpi=300)
            plt.close()
        else:
            plt.show()


def difference_func(x, y, mx, mn):
    """
    Normalize RTs to [0,1] and compute difference
    """
    x = (x - mn) / (mx - mn)
    y = (y - mn) / (mx - mn)
    return x - y


def heat_mapper(data, *, n=None):
    """
    Compute RT-based dissimilarity heatmaps.
    """
    heat_maps = {}
    relations = sorted(data["relation"].unique())
    isis = sorted(data["isi"].unique())

    for isi in isis:
        superset = data[data["isi"] == isi]

        for relation in relations:
            subset = superset[superset["relation"] == relation].reset_index(drop=True)
            if subset.empty:
                continue

            mx = subset["RT"].max()
            mn = subset["RT"].min()

            subset = subset[: n if n is not None else len(subset)]
            n_eff = len(subset)
            if n_eff < 2:
                continue

            heat_map = np.zeros((n_eff, n_eff))

            for i in range(n_eff):
                irt = subset.loc[i, "RT"]
                if pd.isna(irt):
                    continue

                for j in range(i):
                    jrt = subset.loc[j, "RT"]
                    if pd.isna(jrt):
                        continue

                    d = difference_func(float(irt), float(jrt), mx, mn)
                    heat_map[i, j] = d
                    heat_map[j, i] = d

            heat_maps[(isi, relation)] = heat_map

    return heat_maps


def build_meta(data):
    """
    Cache relations, isis, and minimal trial info for bootstrapping.
    trial_cache[(isi, relation)] = [(prime, target), ...]
    """
    relations = sorted(data["relation"].unique())
    isis = sorted(data["isi"].unique())

    trial_cache = {}
    for isi in isis:
        for relation in relations:
            subset = data[
                (data["isi"] == isi) &
                (data["relation"] == relation)
            ]
            trial_cache[(isi, relation)] = list(
                subset[["prime", "target"]].itertuples(index=False, name=None)
            )

    return {
        "relations": relations,
        "isis": isis,
        "trial_cache": trial_cache
    }


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("data", type=str, help="data file")
    arg_parser.add_argument("--heatmapdir", default="D:/heatmaps")
    args = arg_parser.parse_args()

    meta_file = "rt_meta.pkl"
    prefix = "nam" if "nam" in args.data else "ldt"

    if os.path.exists(meta_file):
        # Load cached metadata
        with open(meta_file, "rb") as f:
            meta = pickle.load(f)

        # Load RTs only (still needed for heatmaps)
        data = parse_data(
            args.data,
            extract_dict={
                "prime": "prime",
                "target": "target",
                "RT": "RT",
                "isi": "isi",
                "relation" : "relation"
            }
        )

    else:
        # Full parse (first run only)
        data = parse_data(
            args.data,
            extract_dict={
                "prime": "prime",
                "target": "target",
                "RT": "RT",
                "isi": "isi",
                "relation": "relation"
            }
        )

        meta = build_meta(data)
        with open(meta_file, "wb") as f:
            pickle.dump(meta, f)


    heat_maps = heat_mapper(data=data, n=500)

    os.makedirs(args.heatmapdir, exist_ok=True)

    heatmap_path = os.path.join(
        args.heatmapdir,
        f"{prefix}_RT_dissimilarity.npz"
    )

    np.savez_compressed(
        heatmap_path,
        **{f"{k[0]}_{k[1]}": v for k, v in heat_maps.items()}
    )

    plot_heatmaps(
        heat_maps=heat_maps,
        save_dir=args.heatmapdir,
        figsize=(6, 5),
        cmap="viridis"
    )
