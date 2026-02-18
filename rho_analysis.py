import json
import argparse
import os
from typing import List
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import spearmanr

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "axes.linewidth": 1.2,
    "lines.linewidth": 1.5,
    "lines.markersize": 3,
})


# ============================================================
# Utils
# ============================================================

def latex_escape(s):
    return str(s).replace("_", "\\_")


def component_sort_key(c):
    if c == "word_embeddings":
        return 0
    if c.startswith("encoder_layer_"):
        return int(c.split("_")[-1])
    return 999


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--rhos", default="rhos.json")
    parser.add_argument("--conditions", type=List[str], nargs="+", default=["nam", "ldt"])
    parser.add_argument("--graphs", type=str, default="rho_graphs")

    parser.add_argument("--a", type=str, default="associated",
                        help="First relation label")
    parser.add_argument("--b", type=str, default="similar",
                        help="Second relation label")

    args = parser.parse_args()

    assert os.path.exists(args.rhos)
    os.makedirs(args.graphs, exist_ok=True)

    ISIS = [50, 1050]

    with open(args.rhos, "r") as f:
        rhos = json.load(f)

    # ------------------------------------------------------------
    # Container per dataset
    # ------------------------------------------------------------
    conditions = {
        cond: {
            "components": [],
            "rho_dict": {},
            "primeconditions": set()
        }
        for cond in args.conditions
    }

    # ------------------------------------------------------------
    # Load JSON (bootstrap-aware)
    # ------------------------------------------------------------
    for entry in rhos.values():

        dataset = entry["dataset"]
        model = entry["model"]
        component = entry["component"]
        relation = entry["relation"]
        isi = entry["isi"]
        bs = entry.get("bootstrap")

        if dataset not in conditions:
            continue

        conditions[dataset]["primeconditions"].add(relation)

        if component not in conditions[dataset]["components"]:
            conditions[dataset]["components"].append(component)

        rho_store = conditions[dataset]["rho_dict"]

        rho_store.setdefault(model, {})
        rho_store[model].setdefault(component, {})
        rho_store[model][component].setdefault(relation, {})
        rho_store[model][component][relation].setdefault(isi, [])

        rho_store[model][component][relation][isi].append({
            "rho": entry["spearman_rho"],
            "p": entry["p_value"],
            "n": entry["n"],
            "bs": bs
        })

    # ============================================================
    # INDEPENDENT RELATION PLOTS
    # ============================================================

    for dataset, data in conditions.items():

        rho_dict = data["rho_dict"]
        components = sorted(data["components"], key=component_sort_key)
        models = sorted(rho_dict.keys())

        for relation in [args.a, args.b]:

            for isi in ISIS:

                fig, ax = plt.subplots(figsize=(11, 5))
                x = np.arange(len(components))
                plotted_any = False

                for model in models:

                    means = []
                    lower_err = []
                    upper_err = []

                    for comp in components:

                        entries = (
                            rho_dict
                            .get(model, {})
                            .get(comp, {})
                            .get(relation, {})
                            .get(isi, [])
                        )

                        if not entries:
                            means.append(np.nan)
                            lower_err.append(np.nan)
                            upper_err.append(np.nan)
                            continue

                        values = np.array([e["rho"] for e in entries])

                        mean = np.mean(values)
                        lower = np.percentile(values, 2.5)
                        upper = np.percentile(values, 97.5)

                        means.append(mean)
                        lower_err.append(mean - lower)
                        upper_err.append(upper - mean)

                    if not any(np.isfinite(means)):
                        continue

                    plotted_any = True

                    ax.errorbar(
                        x,
                        means,
                        yerr=[lower_err, upper_err],
                        fmt='o-',
                        capsize=4,
                        label=model
                    )

                if not plotted_any:
                    plt.close(fig)
                    continue

                ax.axhline(0, linestyle="--", linewidth=1)
                ax.set_xticks(x)
                ax.set_xticklabels(components, rotation=45, ha="right")
                ax.set_ylabel("$\\rho$")
                ax.set_xlabel("Component / encoder layer")
                ax.set_title(f"{relation.capitalize()}, ISI {isi} ms")
                ax.set_ylim(-0.25, 0.25)

                ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
                fig.tight_layout(rect=[0, 0, 0.82, 1])

                fname = f"{dataset}_{relation}_isi{isi}.png"
                fig.savefig(os.path.join(args.graphs, fname), bbox_inches="tight")
                plt.close(fig)

                print(f"Saved {fname}")

    # ============================================================
    # DIFFERENCE PLOTS (args.a - args.b)
    # ============================================================

    for dataset, data in conditions.items():

        rho_dict = data["rho_dict"]
        components = sorted(data["components"], key=component_sort_key)
        models = sorted(rho_dict.keys())

        for isi in ISIS:

            fig, ax = plt.subplots(figsize=(11, 5))
            x = np.arange(len(components))
            plotted_any = False

            for model in models:

                means = []
                lower_err = []
                upper_err = []

                for comp in components:

                    b_entries = (
                        rho_dict
                        .get(model, {})
                        .get(comp, {})
                        .get(args.b, {})
                        .get(isi, [])
                    )

                    a_entries = (
                        rho_dict
                        .get(model, {})
                        .get(comp, {})
                        .get(args.a, {})
                        .get(isi, [])
                    )

                    if not a_entries or not b_entries:
                        means.append(np.nan)
                        lower_err.append(np.nan)
                        upper_err.append(np.nan)
                        continue

                    b_dict = {e["bs"]: e["rho"] for e in b_entries}
                    a_dict = {e["bs"]: e["rho"] for e in a_entries}

                    common_bs = sorted(set(a_dict.keys()) & set(b_dict.keys()))

                    if not common_bs:
                        means.append(np.nan)
                        lower_err.append(np.nan)
                        upper_err.append(np.nan)
                        continue

                    diff_vals = np.array([
                        a_dict[b] - b_dict[b]
                        for b in common_bs
                    ])

                    mean = np.mean(diff_vals)
                    lower = np.percentile(diff_vals, 2.5)
                    upper = np.percentile(diff_vals, 97.5)

                    means.append(mean)
                    lower_err.append(mean - lower)
                    upper_err.append(upper - mean)

                if not any(np.isfinite(means)):
                    continue

                plotted_any = True

                ax.errorbar(
                    x,
                    means,
                    yerr=[lower_err, upper_err],
                    fmt='o-',
                    capsize=4,
                    label=model
                )

            if not plotted_any:
                plt.close(fig)
                continue

            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xticks(x)
            ax.set_xticklabels(components, rotation=45, ha="right")
            ax.set_ylim(-0.25, 0.25)

            ax.set_ylabel(
                f"$\\rho$ ({args.a.capitalize()} - {args.b.capitalize()})"
            )

            ax.set_xlabel("Component / encoder layer")
            ax.set_title(
                f"{args.a.capitalize()} − {args.b.capitalize()}, ISI {isi} ms"
            )

            ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False)
            fig.tight_layout(rect=[0, 0, 0.82, 1])

            fname = f"{dataset}_{args.a}_minus_{args.b}_isi{isi}.png"
            fig.savefig(os.path.join(args.graphs, fname), bbox_inches="tight")
            plt.close(fig)

            print(f"Saved {fname}")
