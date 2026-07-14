

import csv
import os
import math
import statistics as stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stats_utils import geo_mean, bootstrap_ci

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

COLORS = {"dp": "#1f77b4", "greedy": "#ff7f0e", "mcts": "#2ca02c",
          "mcts_pw": "#17becf", "shot": "#8c564b", "nmcs": "#e377c2"}


def _read_csv(name):
    path = os.path.join(OUT_DIR, name)
    with open(path) as f:
        return list(csv.DictReader(f))


def _grouped_mean_std(rows, x_key, y_key):
    groups = {}
    for r in rows:
        x = int(r[x_key])
        groups.setdefault(x, []).append(float(r[y_key]))
    xs = sorted(groups)
    means = [stats.mean(groups[x]) for x in xs]
    stds = [stats.stdev(groups[x]) if len(groups[x]) > 1 else 0.0 for x in xs]
    return xs, means, stds


def _grouped_median_range(rows, x_key, y_key):
    groups = {}
    for r in rows:
        x = int(r[x_key])
        groups.setdefault(x, []).append(float(r[y_key]))
    xs = sorted(groups)
    medians = [stats.median(groups[x]) for x in xs]
    mins = [min(groups[x]) for x in xs]
    maxs = [max(groups[x]) for x in xs]
    lower_err = [m - lo for m, lo in zip(medians, mins)]
    upper_err = [hi - m for hi, m in zip(maxs, medians)]
    return xs, medians, [lower_err, upper_err]


def plot_small_scale_cost_ratio():
    rows = _read_csv("small_scale_results.csv")
    xs, gr_med, gr_err = _grouped_median_range(rows, "n_tables", "greedy_ratio")
    _, mc_med, mc_err = _grouped_median_range(rows, "n_tables", "mcts_ratio")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, gr_med, yerr=gr_err, marker="o", label="Greedy / DP-optimal (median, min-max)",
                color=COLORS["greedy"], capsize=4)
    ax.errorbar(xs, mc_med, yerr=mc_err, marker="o", label="MCTS / DP-optimal (median, min-max)",
                color=COLORS["mcts"], capsize=4)
    ax.axhline(1.0, color=COLORS["dp"], linestyle="--", label="DP optimal (ratio = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("Cost ratio vs. DP-optimal (log scale, lower = better)")
    ax.set_title("Plan quality: Greedy can be orders of magnitude worse than optimal;\nMCTS stays near-optimal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig1_cost_ratio_small_scale.png"), dpi=150)
    plt.close(fig)


def plot_small_scale_time():
    rows = _read_csv("small_scale_results.csv")
    xs, dp_t, dp_std = _grouped_mean_std(rows, "n_tables", "dp_time")
    _, gr_t, gr_std = _grouped_mean_std(rows, "n_tables", "greedy_time")
    _, mc_t, mc_std = _grouped_mean_std(rows, "n_tables", "mcts_time")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, dp_t, yerr=dp_std, marker="o", label="Exact bushy-DP (optimal)", color=COLORS["dp"], capsize=4)
    ax.errorbar(xs, gr_t, yerr=gr_std, marker="o", label="Greedy", color=COLORS["greedy"], capsize=4)
    ax.errorbar(xs, mc_t, yerr=mc_std, marker="o", label="MCTS", color=COLORS["mcts"], capsize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("Search time (seconds, log scale)")
    ax.set_title("Search time: Exact bushy-DP's exponential blow-up (O(3^n))\nvs. MCTS/Greedy")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig2_time_small_scale.png"), dpi=150)
    plt.close(fig)


def plot_large_scale_ratio():
    rows = _read_csv("large_scale_results.csv")
    xs, ratio_med, ratio_err = _grouped_median_range(rows, "n_tables", "mcts_vs_greedy_ratio")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, ratio_med, yerr=ratio_err, marker="o", color=COLORS["mcts"], capsize=4,
                label="MCTS cost / Greedy cost (median, min-max)")
    ax.axhline(1.0, color="gray", linestyle="--", label="Greedy baseline (ratio = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined (beyond exact-DP feasibility)")
    ax.set_ylabel("MCTS cost / Greedy cost (log scale, lower = MCTS better)")
    ax.set_title("At large scale (no exact optimum computable): MCTS vs Greedy")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig3_large_scale_ratio.png"), dpi=150)
    plt.close(fig)


def plot_large_scale_time():
    rows = _read_csv("large_scale_results.csv")
    xs, gr_t, gr_std = _grouped_mean_std(rows, "n_tables", "greedy_time")
    _, mc_t, mc_std = _grouped_mean_std(rows, "n_tables", "mcts_time")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, gr_t, yerr=gr_std, marker="o", label="Greedy", color=COLORS["greedy"], capsize=4)
    ax.errorbar(xs, mc_t, yerr=mc_std, marker="o", label="MCTS (budget scales with n)", color=COLORS["mcts"], capsize=4)
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("Search time (seconds, log scale)")
    ax.set_title("MCTS's cost is the price of quality:\nhundreds to ~1300x slower than Greedy, but still seconds not minutes")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig4_large_scale_time.png"), dpi=150)
    plt.close(fig)


def plot_dp_blowup_combined():
    """Combines small-scale DP time (measured) to visually emphasize the
    exponential trend, annotated with the theoretical curve."""
    rows = _read_csv("small_scale_results.csv")
    xs, dp_t, dp_std = _grouped_mean_std(rows, "n_tables", "dp_time")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(xs, dp_t, yerr=dp_std, marker="o", color=COLORS["dp"], capsize=4,
                label="Exact DP, measured")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("DP search time (seconds, log scale)")
    ax.set_title("Exact bushy-DP search time grows exponentially (O(3^n)) with table count")
    ax.grid(alpha=0.3, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig5_dp_exponential_blowup.png"), dpi=150)
    plt.close(fig)


def plot_convergence():
    """One curve per instance size, normalized by that instance's Greedy cost,
    so sizes with wildly different absolute costs can share one axis --
    and it directly shows whether MCTS needs more iterations to converge
    as the search space (table count) grows."""
    rows = _read_csv("convergence_results.csv")
    by_n = {}
    for r in rows:
        n = int(r["n_tables"])
        by_n.setdefault(n, []).append((int(r["iteration"]), float(r["ratio_to_greedy"])))
    for n in by_n:
        by_n[n].sort()

    fig, ax = plt.subplots(figsize=(7, 5))
    palette = ["#2ca02c", "#1f77b4", "#9467bd", "#d62728"]
    for i, n in enumerate(sorted(by_n)):
        iters = [it for it, _ in by_n[n]]
        ratios = [r for _, r in by_n[n]]
        ax.plot(iters, ratios, color=palette[i % len(palette)], label=f"n={n} tables")
    ax.axhline(1.0, color=COLORS["greedy"], linestyle="--", label="Greedy baseline (ratio = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("MCTS iterations (simulations)")
    ax.set_ylabel("Best cost so far / Greedy cost (log scale, lower = better)")
    ax.set_title("MCTS convergence vs. instance size:\nlarger search spaces take longer to beat Greedy")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig6_mcts_convergence.png"), dpi=150)
    plt.close(fig)


def plot_fair_baselines():
    """Compares Greedy / random search / randomized-greedy-restarts / MCTS,
    all given the same rollout budget, to isolate whether MCTS's advantage
    comes from the UCB-guided tree search itself or just from spending more
    compute than Greedy. Regret = cost / best-cost-found-among-the-four for
    that instance (so it stays comparable across n even without a DP
    optimum at large n)."""
    rows = _read_csv("fair_baselines_results.csv")
    methods = [
        ("greedy_regret", "Greedy", COLORS["greedy"]),
        ("random_search_regret", "Random search (same budget)", "#7f7f7f"),
        ("randomized_greedy_restarts_regret", "Randomized-greedy restarts (same budget)", "#9467bd"),
        ("mcts_regret", "MCTS", COLORS["mcts"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in methods:
        xs, med, err = _grouped_median_range(rows, "n_tables", key)
        ax.errorbar(xs, med, yerr=err, marker="o", label=label, color=color, capsize=4)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="Best found (regret = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("Cost / best-cost-found-for-instance (log scale, lower = better)")
    ax.set_title("Same rollout budget for all methods:\nMCTS's tree-search edge holds at small/medium n, but narrows by n=25")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig8_fair_baselines.png"), dpi=150)
    plt.close(fig)


def plot_p_greedy_ablation():
    rows = _read_csv("p_greedy_ablation_results.csv")
    groups = {}
    for r in rows:
        p = float(r["p_greedy"])
        groups.setdefault(p, []).append(float(r["mcts_vs_greedy_ratio"]))
    ps = sorted(groups)
    medians = [stats.median(groups[p]) for p in ps]
    mins = [min(groups[p]) for p in ps]
    maxs = [max(groups[p]) for p in ps]
    lower = [m - lo for m, lo in zip(medians, mins)]
    upper = [hi - m for hi, m in zip(maxs, medians)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ps, medians, yerr=[lower, upper], marker="o", color=COLORS["mcts"],
                capsize=4, label="MCTS / Greedy (median, min-max)")
    ax.axhline(1.0, color=COLORS["greedy"], linestyle="--", label="Greedy baseline (ratio = 1)")
    ax.set_xlabel("p_greedy (probability the rollout picks the myopically cheapest merge)")
    ax.set_ylabel("MCTS cost / Greedy cost (log scale, lower = better)")
    ax.set_yscale("log")
    ax.set_title("Rollout policy ablation (n=20 tables, warm-start disabled):\np_greedy has a modest effect here -- even p_greedy=0\nstill beats Greedy by ~10^5x")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig9_p_greedy_ablation.png"), dpi=150)
    plt.close(fig)


def plot_summary_bar():
    """A single 'headline' figure: geometric mean cost ratio vs DP-optimal
    (small scale) side by side for Greedy and MCTS, with 95% bootstrap
    confidence intervals. Geometric mean (not arithmetic) because ratios
    are multiplicative and span many orders of magnitude -- an arithmetic
    mean would be dominated by a single extreme outlier."""
    rows = _read_csv("small_scale_results.csv")
    gr_ratios = [float(r["greedy_ratio"]) for r in rows]
    mc_ratios = [float(r["mcts_ratio"]) for r in rows]

    gr_mean = geo_mean(gr_ratios)
    mc_mean = geo_mean(mc_ratios)
    gr_lo, gr_hi = bootstrap_ci(gr_ratios)
    mc_lo, mc_hi = bootstrap_ci(mc_ratios)

    fig, ax = plt.subplots(figsize=(5, 5))
    methods = ["Greedy", "MCTS"]
    means = [gr_mean, mc_mean]
    err_lo = [gr_mean - gr_lo, mc_mean - mc_lo]
    err_hi = [gr_hi - gr_mean, mc_hi - mc_mean]
    bars = ax.bar(methods, means, color=[COLORS["greedy"], COLORS["mcts"]],
                   yerr=[err_lo, err_hi], capsize=6)
    ax.axhline(1.0, color=COLORS["dp"], linestyle="--", label="DP optimal")
    ax.set_yscale("log")
    ax.set_ylabel("Geometric mean cost ratio vs. DP-optimal (log scale)")
    ax.set_title("Overall plan quality, with 95% bootstrap CI\n(all small-scale instances, n=4..14)")
    ax.legend()
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, m, f"{m:.2f}x",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig7_summary_bar.png"), dpi=150)
    plt.close(fig)


def plot_progressive_widening():
    """Does capping the root's branching factor (Progressive Widening) close
    the gap seen in fig8 at n=25, where randomized-greedy-restarts beat
    vanilla MCTS at equal budget?"""
    rows = _read_csv("progressive_widening_results.csv")
    methods = [
        ("greedy_regret", "Greedy", COLORS["greedy"]),
        ("randomized_greedy_restarts_regret", "Randomized-greedy restarts (same budget)", "#9467bd"),
        ("mcts_regret", "MCTS (vanilla)", COLORS["mcts"]),
        ("mcts_pw_regret", "MCTS + Progressive Widening", COLORS["mcts_pw"]),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    for key, label, color in methods:
        xs, med, err = _grouped_median_range(rows, "n_tables", key)
        ax.errorbar(xs, med, yerr=err, marker="o", label=label, color=color, capsize=4)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="Best found (regret = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("Cost / best-cost-found-for-instance (log scale, lower = better)")
    ax.set_title("Progressive Widening at equal budget:\ndoes capping root branching narrow MCTS's n=25 gap vs. Table 5.4?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig10_progressive_widening.png"), dpi=150)
    plt.close(fig)


def plot_mcts_benchmark():
    """Extends fig8's fair-budget comparison (same instances) with two more
    budget-constrained methods: SHOT (Sequential Halving applied to Trees)
    and NMCS (Nested Monte Carlo Search)."""
    rows = _read_csv("mcts_benchmark_results.csv")
    methods = [
        ("greedy_regret", "Greedy", COLORS["greedy"]),
        ("random_search_regret", "Random search (same budget)", "#7f7f7f"),
        ("randomized_greedy_restarts_regret", "Randomized-greedy restarts (same budget)", "#9467bd"),
        ("mcts_regret", "MCTS", COLORS["mcts"]),
        ("shot_regret", "SHOT (Sequential Halving)", COLORS["shot"]),
        ("nmcs_regret", "NMCS (level 1)", COLORS["nmcs"]),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for key, label, color in methods:
        xs, med, err = _grouped_median_range(rows, "n_tables", key)
        ax.errorbar(xs, med, yerr=err, marker="o", label=label, color=color, capsize=4)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="Best found (regret = 1)")
    ax.set_yscale("log")
    ax.set_xlabel("Number of tables joined")
    ax.set_ylabel("Cost / best-cost-found-for-instance (log scale, lower = better)")
    ax.set_title("Six budget-constrained methods, same instances as Table 5.4:\nis the n=25 gap specific to UCB1/MCTS, or general?")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "fig11_mcts_benchmark.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    plot_small_scale_cost_ratio()
    plot_small_scale_time()
    plot_large_scale_ratio()
    plot_large_scale_time()
    plot_dp_blowup_combined()
    plot_convergence()
    plot_summary_bar()
    plot_fair_baselines()
    plot_p_greedy_ablation()
    plot_progressive_widening()
    plot_mcts_benchmark()
    print("All figures saved to outputs/")
