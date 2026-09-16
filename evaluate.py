"""
evaluate.py
-----------
Offline evaluation script for the report / rubric items 2 & 3 (model
development, evaluation metrics, algorithm comparison, dimensionality
reduction). Produces:

  1. results/comparison_table.csv
       Win rate of the full MirrorAgent (Q-learning ensemble) vs. a pure
       random-move baseline vs. each single predictor used alone, against
       every simulated human archetype, averaged over many trials.

  2. results/winrate_curves.png
       Win-rate-over-time learning curves for MirrorAgent vs each archetype,
       showing the agent adapting live and beating the 33% random baseline.

  3. results/pca_player_types.png
       PCA (6 behavioral features -> 2 components) of many simulated game
       sessions across all archetypes, showing that different play "styles"
       genuinely occupy separable regions of behavioral-feature space --
       this is the justification for using PCA in this project: the state/
       feature space (per-session behavioral statistics) is multi-dimensional
       and PCA reveals real structure in it, rather than being applied for
       its own sake.

Run with:  python evaluate.py
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from predictors import build_predictor_bank
from simulate_players import ARCHETYPES
from analysis import behavioral_features, fit_pca, run_game, exact_binomial_pvalue, FEATURE_COLUMNS

random.seed(42)
np.random.seed(42)

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

N_ROUNDS = 200
N_TRIALS = 15
ALPHAS = [0.1, 0.3, 0.5]
EPSILONS = [0.05, 0.15, 0.3]


def main():
    predictor_names = [p.name for p in build_predictor_bank()]
    agent_kinds = ["mirror", "ucb1", "random"] + predictor_names

    # ---------- 1. comparison table: win rate per agent kind per archetype ----------
    rows = []
    curves_for_plot = {}  # archetype -> list of win_rate_curve (mirror only, for plotting)

    for archetype_name, archetype_fn in ARCHETYPES.items():
        curves_for_plot[archetype_name] = []
        for agent_kind in agent_kinds:
            trial_final_rates = []
            for trial in range(N_TRIALS):
                curve, _ = run_game(agent_kind, archetype_fn, N_ROUNDS)
                trial_final_rates.append(curve[-1])
                if agent_kind == "mirror":
                    curves_for_plot[archetype_name].append(curve)
            rows.append(
                {
                    "archetype": archetype_name,
                    "agent": agent_kind,
                    "mean_win_rate": np.mean(trial_final_rates),
                    "std_win_rate": np.std(trial_final_rates),
                }
            )
        print(f"done: {archetype_name}")

    results_df = pd.DataFrame(rows)
    results_df.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)
    print("\n=== Win rate comparison (mean over {} trials, {} rounds each) ===".format(N_TRIALS, N_ROUNDS))
    print(results_df.pivot(index="agent", columns="archetype", values="mean_win_rate").round(3))

    # ---------- 1b. statistical significance of MirrorAgent's results ----------
    significance_rows = []
    for archetype_name, curves in curves_for_plot.items():
        for curve in curves:
            wins = int(round(curve[-1] * N_ROUNDS))
            alternative = "greater" if archetype_name != "random" else "two-sided"
            p_value = exact_binomial_pvalue(wins, N_ROUNDS, alternative=alternative)
            significance_rows.append({
                "archetype": archetype_name,
                "wins": wins,
                "rounds": N_ROUNDS,
                "win_rate": curve[-1],
                "test": f"exact binomial ({alternative}) vs 33%",
                "p_value": p_value,
                "significant_at_0.05": p_value < 0.05,
            })
    significance_df = pd.DataFrame(significance_rows)
    significance_df.to_csv(os.path.join(OUT_DIR, "significance_tests.csv"), index=False)
    significance_summary = significance_df.groupby("archetype").agg(
        mean_win_rate=("win_rate", "mean"),
        min_p_value=("p_value", "min"),
        significant_trials=("significant_at_0.05", "sum"),
        trials=("p_value", "size"),
    ).reset_index()
    significance_summary.to_csv(os.path.join(OUT_DIR, "significance_summary.csv"), index=False)

    # ---------- 1c. α/ε sensitivity analysis for Q-learning ----------
    sensitivity_rows = []
    for alpha in ALPHAS:
        for epsilon in EPSILONS:
            config = {
                "alpha": alpha,
                "epsilon": epsilon,
                "epsilon_start": epsilon,
                "epsilon_min": epsilon,
                "epsilon_decay": 1.0,
            }
            for archetype_name, archetype_fn in ARCHETYPES.items():
                rates = [
                    run_game("mirror", archetype_fn, N_ROUNDS, config)[0][-1]
                    for _ in range(N_TRIALS)
                ]
                sensitivity_rows.append({
                    "alpha": alpha,
                    "epsilon": epsilon,
                    "archetype": archetype_name,
                    "mean_win_rate": np.mean(rates),
                    "std_win_rate": np.std(rates),
                })
    sensitivity_df = pd.DataFrame(sensitivity_rows)
    sensitivity_df.to_csv(os.path.join(OUT_DIR, "hyperparameter_sensitivity.csv"), index=False)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, archetype_name in zip(axes.flat, ARCHETYPES):
        subset = sensitivity_df[sensitivity_df["archetype"] == archetype_name]
        for alpha in ALPHAS:
            line = subset[subset["alpha"] == alpha].sort_values("epsilon")
            ax.plot(line["epsilon"], line["mean_win_rate"], marker="o", label=f"α={alpha}")
        ax.axhline(1 / 3, color="black", linestyle="--", linewidth=1)
        ax.set_title(archetype_name.replace("_", " ").title())
        ax.set_xlabel("ε")
        ax.set_ylabel("Mean win rate")
        ax.set_xticks(EPSILONS)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Q-learning hyperparameter sensitivity (γ = 0.7)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "hyperparameter_sensitivity.png"), dpi=150)
    plt.close(fig)

    # ---------- 2. win-rate learning curves (mirror vs each archetype) ----------
    plt.figure(figsize=(9, 5))
    for archetype_name, curves in curves_for_plot.items():
        avg_curve = np.mean(curves, axis=0)
        plt.plot(avg_curve, label=archetype_name)
    plt.axhline(1 / 3, color="black", linestyle="--", label="random-chance baseline (33%)")
    plt.xlabel("Round")
    plt.ylabel("MirrorAgent win rate (cumulative)")
    plt.title("MirrorAgent learning curves vs. simulated player archetypes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "winrate_curves.png"), dpi=150)
    plt.close()

    # ---------- 3. PCA of behavioral feature space across many sessions ----------
    feature_rows = []
    labels = []
    for archetype_name, archetype_fn in ARCHETYPES.items():
        for trial in range(30):
            _, history = run_game("mirror", archetype_fn, 60)
            feats = behavioral_features(history)
            feature_rows.append(feats)
            labels.append(archetype_name)

    feat_df = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    feat_df["archetype"] = labels
    feat_df.to_csv(os.path.join(OUT_DIR, "behavioral_features.csv"), index=False)

    coords, explained = fit_pca(feature_rows)

    plt.figure(figsize=(7, 6))
    for archetype_name in ARCHETYPES:
        mask = feat_df["archetype"] == archetype_name
        plt.scatter(coords[mask, 0], coords[mask, 1], label=archetype_name, alpha=0.7)
    plt.xlabel(f"PC1 ({explained[0]*100:.1f}% variance)")
    plt.ylabel(f"PC2 ({explained[1]*100:.1f}% variance)")
    plt.title("PCA of behavioral feature space across simulated player archetypes")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_player_types.png"), dpi=150)
    plt.close()

    print("\nSaved: results/comparison_table.csv, results/winrate_curves.png, "
          "results/hyperparameter_sensitivity.csv, results/hyperparameter_sensitivity.png, "
          "results/significance_tests.csv, results/significance_summary.csv, "
          "results/pca_player_types.png, results/behavioral_features.csv")


if __name__ == "__main__":
    main()
