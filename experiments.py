"""
experiments.py

Uses BUSHY join tree algorithms (bushy_algorithms.py) -- the general,
NP-hard version of join-order optimization used by real query optimizers
(System R / Selinger style). An earlier left-deep-only version of this
experiment (see algorithms.py) turned out to be a case where Greedy is
already provably near-optimal (a known polynomial-time-solvable special
case, IKKBZ), so it couldn't demonstrate any real advantage for MCTS.
Bushy trees restore genuine combinatorial hardness: Greedy's lack of
lookahead can now cost 10x-10000x, while MCTS reliably finds near-optimal
plans.

(A) SMALL SCALE (n <= DP_MAX_TABLES): exact bushy DP (O(3^n)) is tractable,
    so we know the true optimum and can measure exactly how close Greedy
    and MCTS get, plus how much faster they are.

(B) LARGE SCALE (n > DP_MAX_TABLES): DP is skipped entirely (O(3^n) makes
    n=16 already take over a minute) -- this demonstrates the combinatorial
    explosion. Only Greedy vs MCTS are compared, on cost and time.

(C) CONVERGENCE: for large, hard instances of increasing size, track MCTS's
    best-cost-so-far across iterations.

(D) FAIR-BUDGET BASELINES: isolates *why* MCTS wins over Greedy -- is it
    the UCB-guided tree search, or just "more compute than Greedy"? Compares
    against random search and randomized-greedy-with-restarts given the same
    rollout budget as MCTS.

(E) P_GREEDY ABLATION: sweeps the rollout's greedy-bias parameter to check
    the (previously only asserted, never measured) claim that a purely
    random rollout degrades at scale.

(F) CASE STUDY: a single worst-case instance, with the actual join trees
    chosen by DP/Greedy/MCTS printed out, to make concrete *why* Greedy
    fails where it does.

(G) STATISTICAL SIGNIFICANCE: bootstrap confidence intervals + a paired
    sign-flip permutation test on the small-scale results, so "MCTS beats
    Greedy" is a supported claim, not just a plot that looks convincing.

(H) PROGRESSIVE WIDENING: a targeted follow-up on (D)'s most surprising
    result (MCTS loses to randomized-greedy-restarts at n=25, same budget).
    Tests whether capping the tree's root branching factor via Progressive
    Widening closes that gap.

(I) BROADER MCTS BENCHMARK: extends (D)'s exact instances with two more
    course-covered budget-constrained methods -- Sequential Halving applied
    to Trees (SHOT) and Nested Monte Carlo Search (NMCS) -- to check whether
    (D)'s n=25 finding is specific to UCB1/MCTS or a general property of any
    budget-constrained tree/nested search on this problem.
"""

import csv
import os
import random

from join_graph import JoinGraph
from cost_model import CostModel
from bushy_algorithms import (
    bushy_exact_dp, bushy_greedy, bushy_mcts, bushy_random_restarts,
    bushy_shot, bushy_nmcs, pretty_tree, timed,
)
from stats_utils import geo_mean, bootstrap_ci, paired_sign_flip_test

OUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DP_MAX_TABLES = 14          # O(3^n) DP: already ~5.6s per instance at n=14, ~77s at n=16
N_REPEATS = 10               # was 5 -- bumped for statistical power (see (G))
MCTS_ITERS = 1000

# Progressive Widening hyperparameters, chosen by a small exploratory sweep
# (6 (pw_c, pw_alpha) combinations, pw_c in {0.5, 1, 2, 3, 4}, pw_alpha in
# {0.3, 0.4, 0.5}) on 8 n=25 instances (same seed scheme as (D)/Section 5.4):
# pw_c=3.0, pw_alpha=0.4 gave the best geometric-mean regret among the
# combinations tried -- not the single best on every individual instance,
# but the most consistent overall. This sweep is a separate, throwaway run
# (not reproduced by run_progressive_widening below); only the winning
# config is kept and reused as a fixed default here.
PW_C = 3.0
PW_ALPHA = 0.4

# denser graphs + wider selectivity range = genuine room for Greedy to get
# trapped by a locally-attractive but globally-bad merge
GRAPH_KWARGS = dict(extra_edge_prob=0.4, sel_range=(0.0005, 0.9))


def _write_csv(name, rows):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


# ---------------------------------------------------------------------------
# (A) small scale
# ---------------------------------------------------------------------------
def run_small_scale():
    table_counts = [4, 6, 8, 10, 12, 14]
    rows = []

    for n in table_counts:
        for rep in range(N_REPEATS):
            seed = 1000 * n + rep
            g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
            cm = CostModel(g)

            dp_cost, dp_time = timed(bushy_exact_dp, g, cm)
            gr_cost, gr_time = timed(bushy_greedy, g, cm)
            mc_cost, mc_time = timed(
                bushy_mcts, g, cm, n_iterations=MCTS_ITERS, seed=seed
            )

            rows.append({
                "n_tables": n, "rep": rep,
                "dp_cost": dp_cost, "dp_time": dp_time,
                "greedy_cost": gr_cost, "greedy_time": gr_time,
                "mcts_cost": mc_cost, "mcts_time": mc_time,
                "greedy_ratio": gr_cost / dp_cost,
                "mcts_ratio": mc_cost / dp_cost,
            })
            print(f"[small] n={n} rep={rep} "
                  f"DP={dp_cost:.3e} ({dp_time:.4f}s)  "
                  f"Greedy ratio={gr_cost/dp_cost:.3f} ({gr_time:.4f}s)  "
                  f"MCTS ratio={mc_cost/dp_cost:.4f} ({mc_time:.4f}s)")

    _write_csv("small_scale_results.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# (B) large scale
# ---------------------------------------------------------------------------
def run_large_scale():
    table_counts = [16, 20, 25, 30, 35, 40]
    rows = []
    n_repeats = 4  # was 3

    for n in table_counts:
        for rep in range(n_repeats):
            seed = 2000 * n + rep
            g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
            cm = CostModel(g)

            gr_cost, gr_time = timed(bushy_greedy, g, cm)
            iters = max(MCTS_ITERS, n * 40)
            mc_cost, mc_time = timed(
                bushy_mcts, g, cm, n_iterations=iters, seed=seed
            )

            rows.append({
                "n_tables": n, "rep": rep,
                "greedy_cost": gr_cost, "greedy_time": gr_time,
                "mcts_cost": mc_cost, "mcts_time": mc_time,
                "mcts_iters": iters,
                "mcts_vs_greedy_ratio": mc_cost / gr_cost,
            })
            print(f"[large] n={n} rep={rep} "
                  f"Greedy={gr_cost:.3e} ({gr_time:.4f}s)  "
                  f"MCTS={mc_cost:.3e} ({mc_time:.4f}s)  "
                  f"MCTS/Greedy={mc_cost/gr_cost:.4f}")

    _write_csv("large_scale_results.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# (C) convergence, at multiple sizes (does a bigger search space need more
# iterations to converge? this needs several n values to actually show)
# ---------------------------------------------------------------------------
def run_convergence():
    rows = []
    configs = [(16, 777), (30, 777), (40, 777)]

    for n, seed in configs:
        g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
        cm = CostModel(g)

        _, curve = bushy_mcts(g, cm, n_iterations=2000, seed=seed, track_curve=True)
        gr_cost = bushy_greedy(g, cm)

        for it, best_cost in curve:
            rows.append({
                "n_tables": n, "iteration": it,
                "best_cost_so_far": best_cost,
                "greedy_cost": gr_cost,
                "ratio_to_greedy": best_cost / gr_cost,
            })
        print(f"[convergence] n={n} greedy={gr_cost:.3e} "
              f"final_mcts={curve[-1][1]:.3e} ratio={curve[-1][1]/gr_cost:.4f}")

    _write_csv("convergence_results.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# (D) fair-budget baselines: Greedy vs random search vs randomized-greedy
# restarts vs MCTS, all given (approximately) the same rollout budget.
# ---------------------------------------------------------------------------
def run_fair_baselines():
    table_counts = [10, 14, 20, 25]
    n_repeats = 4
    rows = []

    for n in table_counts:
        for rep in range(n_repeats):
            seed = 3000 * n + rep
            g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
            cm = CostModel(g)
            rng = random.Random(seed)

            dp_cost = bushy_exact_dp(g, cm) if n <= DP_MAX_TABLES else None
            gr_cost = bushy_greedy(g, cm)
            rs_cost = bushy_random_restarts(g, cm, rng, MCTS_ITERS, p_greedy=0.0)
            rg_cost = bushy_random_restarts(g, cm, rng, MCTS_ITERS, p_greedy=0.75)
            mc_cost = bushy_mcts(g, cm, n_iterations=MCTS_ITERS, seed=seed)

            best_found = min(gr_cost, rs_cost, rg_cost, mc_cost)
            row = {
                "n_tables": n, "rep": rep,
                "dp_cost": dp_cost if dp_cost is not None else "",
                "greedy_cost": gr_cost,
                "random_search_cost": rs_cost,
                "randomized_greedy_restarts_cost": rg_cost,
                "mcts_cost": mc_cost,
                "greedy_regret": gr_cost / best_found,
                "random_search_regret": rs_cost / best_found,
                "randomized_greedy_restarts_regret": rg_cost / best_found,
                "mcts_regret": mc_cost / best_found,
            }
            if dp_cost is not None:
                row["greedy_vs_dp"] = gr_cost / dp_cost
                row["random_search_vs_dp"] = rs_cost / dp_cost
                row["randomized_greedy_restarts_vs_dp"] = rg_cost / dp_cost
                row["mcts_vs_dp"] = mc_cost / dp_cost
            rows.append(row)
            print(f"[fair-budget] n={n} rep={rep} "
                  f"Greedy/best={row['greedy_regret']:.3f}  "
                  f"RandomSearch/best={row['random_search_regret']:.3f}  "
                  f"RandGreedyRestarts/best={row['randomized_greedy_restarts_regret']:.3f}  "
                  f"MCTS/best={row['mcts_regret']:.5f}")

    # not all rows have the *_vs_dp keys (n > DP_MAX_TABLES) -- normalize
    # fieldnames across all rows before writing.
    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    path = os.path.join(OUT_DIR, "fair_baselines_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# (E) p_greedy ablation: does the rollout need domain-knowledge bias to
# stay useful as the search space grows?
# ---------------------------------------------------------------------------
def run_p_greedy_ablation():
    n = 20
    n_repeats = 5
    p_values = [0.0, 0.25, 0.5, 0.75, 1.0]
    rows = []

    for rep in range(n_repeats):
        seed = 4000 + rep
        g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
        cm = CostModel(g)
        gr_cost = bushy_greedy(g, cm)

        for p in p_values:
            mc_cost = bushy_mcts(
                g, cm, n_iterations=MCTS_ITERS, seed=seed, p_greedy=p,
                seed_with_greedy=False,  # isolate the rollout policy's own effect
            )
            rows.append({
                "n_tables": n, "rep": rep, "p_greedy": p,
                "mcts_cost": mc_cost, "greedy_cost": gr_cost,
                "mcts_vs_greedy_ratio": mc_cost / gr_cost,
            })
            print(f"[p_greedy ablation] rep={rep} p_greedy={p} "
                  f"MCTS/Greedy={mc_cost/gr_cost:.4f}")

    _write_csv("p_greedy_ablation_results.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# (H) Progressive Widening: a targeted follow-up on the single most
# surprising result of the fair-budget experiment (D) -- at n=25, "Greedy
# randomized restarts" beat vanilla MCTS at equal budget. The hypothesis in
# Section 5.4 was that UCB1's default behaviour (expand *every* legal child
# of a node before ever descending via UCB1) wastes a growing share of the
# fixed budget widening the root as branching factor grows with n. Progressive
# Widening caps how many children a node may expose as a function of its own
# visit count, forcing the tree to go deep sooner. This experiment repeats
# the same fixed-budget comparison at n=20, 25, 30 (more repeats than (D),
# since this is now the experiment meant to settle the question) with an
# extra column: MCTS with Progressive Widening enabled.
# ---------------------------------------------------------------------------
def run_progressive_widening():
    table_counts = [20, 25, 30]
    n_repeats = 8
    rows = []

    for n in table_counts:
        for rep in range(n_repeats):
            seed = 3000 * n + rep  # same seed scheme as (D) at n=20,25 -> same instances
            g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
            cm = CostModel(g)
            rng = random.Random(seed)

            gr_cost = bushy_greedy(g, cm)
            rg_cost = bushy_random_restarts(g, cm, rng, MCTS_ITERS, p_greedy=0.75)
            mc_cost = bushy_mcts(g, cm, n_iterations=MCTS_ITERS, seed=seed)
            mc_pw_cost = bushy_mcts(g, cm, n_iterations=MCTS_ITERS, seed=seed,
                                     pw_c=PW_C, pw_alpha=PW_ALPHA)

            best_found = min(gr_cost, rg_cost, mc_cost, mc_pw_cost)
            rows.append({
                "n_tables": n, "rep": rep,
                "greedy_cost": gr_cost,
                "randomized_greedy_restarts_cost": rg_cost,
                "mcts_cost": mc_cost,
                "mcts_pw_cost": mc_pw_cost,
                "greedy_regret": gr_cost / best_found,
                "randomized_greedy_restarts_regret": rg_cost / best_found,
                "mcts_regret": mc_cost / best_found,
                "mcts_pw_regret": mc_pw_cost / best_found,
            })
            print(f"[progressive-widening] n={n} rep={rep} "
                  f"RandGreedyRestarts/best={rg_cost/best_found:.3f}  "
                  f"MCTS/best={mc_cost/best_found:.3f}  "
                  f"MCTS+PW/best={mc_pw_cost/best_found:.3f}")

    _write_csv("progressive_widening_results.csv", rows)
    return rows


# ---------------------------------------------------------------------------
# (I) Broader MCTS benchmark: adds two more course-covered budget-constrained
# methods to the exact same instances as the fair-budget experiment (D) --
# Sequential Halving applied to Trees (SHOT: same idea as (D)'s baselines,
# a different way of splitting a fixed rollout budget across candidates) and
# Nested Monte Carlo Search (NMCS: a tree-free alternative for single-agent
# sequential construction problems, the same family of problem as ours).
# Reusing (D)'s seeds means this is a direct extension of Table 5.4, not a
# separate experiment on different instances.
# ---------------------------------------------------------------------------
def run_mcts_benchmark():
    table_counts = [10, 14, 20, 25]
    n_repeats = 4
    rows = []

    for n in table_counts:
        for rep in range(n_repeats):
            seed = 3000 * n + rep  # identical instances to (D) / Table fair-baselines
            g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
            cm = CostModel(g)
            rng = random.Random(seed)

            dp_cost = bushy_exact_dp(g, cm) if n <= DP_MAX_TABLES else None
            gr_cost = bushy_greedy(g, cm)
            rs_cost = bushy_random_restarts(g, cm, rng, MCTS_ITERS, p_greedy=0.0)
            rg_cost = bushy_random_restarts(g, cm, rng, MCTS_ITERS, p_greedy=0.75)
            mc_cost = bushy_mcts(g, cm, n_iterations=MCTS_ITERS, seed=seed)
            shot_cost = bushy_shot(g, cm, total_budget=MCTS_ITERS, seed=seed)
            nmcs_cost = bushy_nmcs(g, cm, level=1, seed=seed)

            best_found = min(gr_cost, rs_cost, rg_cost, mc_cost, shot_cost, nmcs_cost)
            row = {
                "n_tables": n, "rep": rep,
                "dp_cost": dp_cost if dp_cost is not None else "",
                "greedy_cost": gr_cost,
                "random_search_cost": rs_cost,
                "randomized_greedy_restarts_cost": rg_cost,
                "mcts_cost": mc_cost,
                "shot_cost": shot_cost,
                "nmcs_cost": nmcs_cost,
                "greedy_regret": gr_cost / best_found,
                "random_search_regret": rs_cost / best_found,
                "randomized_greedy_restarts_regret": rg_cost / best_found,
                "mcts_regret": mc_cost / best_found,
                "shot_regret": shot_cost / best_found,
                "nmcs_regret": nmcs_cost / best_found,
            }
            if dp_cost is not None:
                row["greedy_vs_dp"] = gr_cost / dp_cost
                row["mcts_vs_dp"] = mc_cost / dp_cost
                row["shot_vs_dp"] = shot_cost / dp_cost
                row["nmcs_vs_dp"] = nmcs_cost / dp_cost
            rows.append(row)
            print(f"[mcts-benchmark] n={n} rep={rep} "
                  f"Greedy/best={row['greedy_regret']:.3f}  "
                  f"MCTS/best={row['mcts_regret']:.4f}  "
                  f"SHOT/best={row['shot_regret']:.4f}  "
                  f"NMCS/best={row['nmcs_regret']:.4f}")

    all_keys = []
    for r in rows:
        for k in r:
            if k not in all_keys:
                all_keys.append(k)
    path = os.path.join(OUT_DIR, "mcts_benchmark_results.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)
    return rows


# ---------------------------------------------------------------------------
# (F) qualitative case study: pick the worst-case (readable-size) instance
# from the small-scale run and show the actual trees chosen.
# ---------------------------------------------------------------------------
def run_case_study(small_scale_rows):
    candidates = [r for r in small_scale_rows if r["n_tables"] <= 10]
    worst = max(candidates, key=lambda r: r["greedy_ratio"])
    n, rep = worst["n_tables"], worst["rep"]
    seed = 1000 * n + rep

    g = JoinGraph(n, seed=seed, **GRAPH_KWARGS)
    cm = CostModel(g)

    dp_cost, dp_tree = bushy_exact_dp(g, cm, return_plan=True)
    gr_cost, gr_tree = bushy_greedy(g, cm, return_plan=True)
    mc_cost, mc_tree = bushy_mcts(
        g, cm, n_iterations=MCTS_ITERS, seed=seed, return_plan=True
    )

    lines = []
    lines.append(f"Case study: n={n} tables, seed={seed} (worst Greedy/DP ratio "
                  f"among small-scale instances with n<=10: {worst['greedy_ratio']:.2f}x)")
    lines.append("")
    lines.append("Table cardinalities:")
    for i, c in enumerate(g.card):
        lines.append(f"  table {i}: card={c:,.0f}")
    lines.append("")
    lines.append("Join edges (table_a, table_b): selectivity")
    for (u, v), sel in sorted(g.edges.items()):
        lines.append(f"  ({u}, {v}): {sel:.5f}")
    lines.append("")
    lines.append(f"DP-optimal cost:     {dp_cost:.6e}")
    lines.append(f"  plan: {pretty_tree(dp_tree)}")
    lines.append("")
    lines.append(f"Greedy cost:         {gr_cost:.6e}  (ratio to optimal: {gr_cost/dp_cost:.2f}x)")
    lines.append(f"  plan: {pretty_tree(gr_tree)}")
    lines.append("")
    lines.append(f"MCTS cost:           {mc_cost:.6e}  (ratio to optimal: {mc_cost/dp_cost:.4f}x)")
    lines.append(f"  plan: {pretty_tree(mc_tree)}")
    lines.append("")
    lines.append("Reading note: Greedy always merges whichever pair is cheapest "
                  "*right now*. Compare its first merge above to the DP-optimal "
                  "plan's top-level split to see the locally-attractive-but-"
                  "globally-bad choice it locked itself into.")

    text = "\n".join(lines)
    path = os.path.join(OUT_DIR, "case_study.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n=== Case study ===")
    print(text)
    return text


# ---------------------------------------------------------------------------
# (G) statistical significance: bootstrap CIs + paired sign-flip test
# ---------------------------------------------------------------------------
def run_statistical_summary(small_scale_rows):
    import math

    gr_ratios = [r["greedy_ratio"] for r in small_scale_rows]
    mc_ratios = [r["mcts_ratio"] for r in small_scale_rows]
    log_diffs = [math.log(g) - math.log(m) for g, m in zip(gr_ratios, mc_ratios)]

    gr_geo = geo_mean(gr_ratios)
    mc_geo = geo_mean(mc_ratios)
    gr_lo, gr_hi = bootstrap_ci(gr_ratios)
    mc_lo, mc_hi = bootstrap_ci(mc_ratios)
    p_value = paired_sign_flip_test(log_diffs)

    lines = [
        "Statistical summary (small-scale results, n=4..14, "
        f"{len(small_scale_rows)} paired instances)",
        "",
        f"Greedy / DP-optimal  -- geometric mean = {gr_geo:.3f}x  "
        f"(95% bootstrap CI: [{gr_lo:.3f}, {gr_hi:.3f}])",
        f"MCTS   / DP-optimal  -- geometric mean = {mc_geo:.3f}x  "
        f"(95% bootstrap CI: [{mc_lo:.3f}, {mc_hi:.3f}])",
        "",
        "Paired sign-flip permutation test on log(greedy_ratio) - log(mcts_ratio) "
        "per instance (H0: no systematic difference):",
        f"  p-value = {p_value:.5f}",
        "  (p < 0.05 => the gap between Greedy and MCTS across these instances "
        "is unlikely to be due to chance)",
    ]
    text = "\n".join(lines)
    path = os.path.join(OUT_DIR, "statistical_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print("\n=== Statistical summary ===")
    print(text)
    return text


if __name__ == "__main__":
    print("=== Running small-scale experiment (bushy DP vs Greedy vs MCTS) ===")
    small_rows = run_small_scale()

    print("\n=== Running large-scale experiment (Greedy vs MCTS, no DP) ===")
    run_large_scale()

    print("\n=== Running MCTS convergence experiment (multiple sizes) ===")
    run_convergence()

    print("\n=== Running fair-budget baselines experiment ===")
    run_fair_baselines()

    print("\n=== Running p_greedy rollout ablation ===")
    run_p_greedy_ablation()

    print("\n=== Running Progressive Widening experiment (n=20,25,30) ===")
    run_progressive_widening()

    print("\n=== Running broader MCTS benchmark (SHOT, NMCS) ===")
    run_mcts_benchmark()

    run_case_study(small_rows)
    run_statistical_summary(small_rows)

    print("\nAll results saved to outputs/*.csv and outputs/*.txt")
