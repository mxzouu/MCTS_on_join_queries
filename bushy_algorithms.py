"""
bushy_algorithms.py

The GENERAL join-order problem (as in real optimizers like System R /
Selinger) allows BUSHY join trees: at each step you may merge any two
already-built partial results, not just extend one growing chain by a
single table. This is what makes join-order optimization genuinely
NP-hard -- the left-deep/chain restriction used in `algorithms.py` happens
to be polynomial-time solvable for tree-shaped query graphs under the
"sum of intermediate sizes" cost model (a classical result, see the IKKBZ
algorithm, Krishnamurthy/Boral/Zaniolo 1986), which is exactly why Greedy
and MCTS looked identical there: there was no real combinatorial hardness
to exploit.

State representation: a "forest" = a frozenset of frozensets, each inner
frozenset being one already-built partial join group. Initially it's n
singleton groups; a "merge" action combines two connected groups into one;
the process ends when a single group remains.

1. bushy_exact_dp   - O(3^n) subset-DP (Selinger-style), the true optimum.
                      Only tractable for small n (this blows up much faster
                      than the O(2^n) left-deep DP).
2. bushy_greedy     - agglomerative: repeatedly merge whichever connected
                      pair of groups is cheapest right now (myopic).
3. bushy_mcts       - MCTS over the space of merge sequences.
"""

import math
import random
import time


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _connected(graph, set_a, set_b):
    for a in set_a:
        if graph.adj[a] & set_b:
            return True
    return False


def _initial_forest(n):
    return frozenset(frozenset([i]) for i in range(n))


def _legal_merges(graph, forest):
    groups = list(forest)
    pairs = []
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            if _connected(graph, groups[i], groups[j]):
                pairs.append((groups[i], groups[j]))
    return pairs


def _apply_merge(forest, a, b):
    return (forest - {a, b}) | {a | b}


def _tree_from_merges(n, merges):
    """Replays an ordered list of (a, b, cost) group-merges into a nested-tuple
    binary tree (leaves = table indices) for display/case-study purposes."""
    node = {frozenset([i]): i for i in range(n)}
    for a, b, _ in merges:
        node[a | b] = (node[a], node[b])
    return node[frozenset(range(n))]


def _dp_tree(mask, dp_split):
    if bin(mask).count("1") == 1:
        return mask.bit_length() - 1
    sub, other = dp_split[mask]
    return (_dp_tree(sub, dp_split), _dp_tree(other, dp_split))


def pretty_tree(node):
    if isinstance(node, int):
        return str(node)
    left, right = node
    return f"({pretty_tree(left)} JOIN {pretty_tree(right)})"


# ---------------------------------------------------------------------------
# 1. Exact bushy DP  (O(3^n))
# ---------------------------------------------------------------------------
def bushy_exact_dp(graph, cost_model, return_plan=False):
    n = graph.n
    FULL = (1 << n) - 1
    dp_cost = {}
    dp_split = {}

    for i in range(n):
        dp_cost[1 << i] = graph.card[i]  # base scan cost

    masks_by_popcount = [[] for _ in range(n + 1)]
    for mask in range(1, 1 << n):
        masks_by_popcount[bin(mask).count("1")].append(mask)

    for popcount in range(2, n + 1):
        for mask in masks_by_popcount[popcount]:
            members = [i for i in range(n) if mask & (1 << i)]
            best_cost, best_split = None, None
            # enumerate all non-empty proper submasks of `mask`
            sub = (mask - 1) & mask
            while sub > 0:
                other = mask ^ sub
                if sub < other:  # avoid duplicate (sub,other)/(other,sub) work
                    sub = (sub - 1) & mask
                    continue
                if sub in dp_cost and other in dp_cost:
                    left = [i for i in range(n) if sub & (1 << i)]
                    right = [i for i in range(n) if other & (1 << i)]
                    if _connected(graph, set(left), set(right)):
                        jcost = cost_model.join_pair_cost(set(left), set(right))
                        total = dp_cost[sub] + dp_cost[other] + jcost
                        if best_cost is None or total < best_cost:
                            best_cost, best_split = total, (sub, other)
                sub = (sub - 1) & mask
            if best_cost is not None:
                dp_cost[mask] = best_cost
                dp_split[mask] = best_split

    cost = dp_cost.get(FULL, float("inf"))
    if return_plan:
        tree = _dp_tree(FULL, dp_split) if FULL in dp_split else None
        return cost, tree
    return cost


# ---------------------------------------------------------------------------
# 2. Agglomerative greedy
# ---------------------------------------------------------------------------
def bushy_greedy(graph, cost_model, return_plan=False):
    forest = _initial_forest(graph.n)
    total_cost = sum(graph.card[i] for i in range(graph.n))  # scan costs
    merges = [] if return_plan else None

    while len(forest) > 1:
        pairs = _legal_merges(graph, forest)
        if not pairs:
            # disconnected fallback: merge any two groups
            groups = list(forest)
            pairs = [(groups[0], groups[1])]
        best_pair, best_cost = None, None
        for (a, b) in pairs:
            c = cost_model.join_pair_cost(a, b)
            if best_cost is None or c < best_cost:
                best_pair, best_cost = (a, b), c
        a, b = best_pair
        forest = _apply_merge(forest, a, b)
        total_cost += best_cost
        if return_plan:
            merges.append((a, b, best_cost))

    if return_plan:
        return total_cost, _tree_from_merges(graph.n, merges)
    return total_cost


# ---------------------------------------------------------------------------
# 2b. Fair-budget baselines: random-restart search
#
# Both Greedy and MCTS get compared against these to isolate *why* MCTS wins:
# is it the UCB-guided tree search itself, or just "more compute than Greedy"?
# Both baselines spend exactly `n_restarts` full rollouts, the same order of
# magnitude of work as `n_iterations` MCTS iterations (each MCTS iteration
# also performs one rollout, plus lightweight tree bookkeeping) -- a fair,
# if approximate, budget match.
#   - p_greedy=0.0  -> pure random search (no domain knowledge at all)
#   - p_greedy=0.75 -> randomized-greedy restarts (same rollout bias MCTS
#                      uses by default) -- the real control: if this alone
#                      matches MCTS, the tree search adds nothing.
# ---------------------------------------------------------------------------
def bushy_random_restarts(graph, cost_model, rng, n_restarts, p_greedy):
    scan_cost = sum(graph.card[i] for i in range(graph.n))
    root_forest = _initial_forest(graph.n)
    best_cost = float("inf")
    for _ in range(n_restarts):
        cost = _rollout(graph, cost_model, root_forest, scan_cost, rng, p_greedy)
        if cost < best_cost:
            best_cost = cost
    return best_cost


# ---------------------------------------------------------------------------
# 3. MCTS over merge sequences
# ---------------------------------------------------------------------------
class _Node:
    __slots__ = ("forest", "children", "visits", "total_value", "untried")

    def __init__(self, forest, legal_pairs):
        self.forest = forest
        self.children = {}
        self.visits = 0
        self.total_value = 0.0
        self.untried = list(legal_pairs)


def _rollout(graph, cost_model, forest, running_cost, rng, p_greedy=0.75, record=False):
    forest = set(forest)
    total = running_cost
    merges = [] if record else None
    while len(forest) > 1:
        pairs = _legal_merges(graph, forest)
        if not pairs:
            groups = list(forest)
            pairs = [(groups[0], groups[1])]
        if len(pairs) > 1 and rng.random() < p_greedy:
            a, b = min(pairs, key=lambda p: cost_model.join_pair_cost(p[0], p[1]))
        else:
            a, b = rng.choice(pairs)
        c = cost_model.join_pair_cost(a, b)
        forest = (forest - {a, b}) | {a | b}
        total += c
        if record:
            merges.append((a, b, c))
    if record:
        return total, merges
    return total


def bushy_mcts(graph, cost_model, n_iterations=1500, c_ucb=1.4, seed=None,
               p_greedy=0.75, seed_with_greedy=True, track_curve=False,
               return_plan=False, pw_c=None, pw_alpha=0.5):
    """pw_c/pw_alpha enable Progressive Widening: instead of expanding *all*
    of a node's legal children before ever using UCB1 (the default,
    unbounded-widening behaviour when pw_c=None), a node visited N times may
    only have up to k(N) = ceil(pw_c * (N+1)**pw_alpha) children expanded.
    This caps how much of the fixed iteration budget is spent widening the
    root breadth-first before the tree is allowed to go deep -- directly
    targeting the failure mode discussed in Section 5.4 of the report, where
    a growing branching factor at the root was hypothesized to eat an
    increasing share of the budget."""
    rng = random.Random(seed)
    n = graph.n
    scan_cost = sum(graph.card[i] for i in range(n))

    root_forest = _initial_forest(n)
    root = _Node(root_forest, _legal_merges(graph, root_forest))

    best_cost = float("inf")
    best_tree = None
    if seed_with_greedy:
        if return_plan:
            best_cost, best_tree = bushy_greedy(graph, cost_model, return_plan=True)
        else:
            best_cost = bushy_greedy(graph, cost_model)

    curve = []

    def _can_widen(node):
        if not node.untried:
            return False
        if pw_c is None:
            return True
        limit = max(1, math.ceil(pw_c * (node.visits + 1) ** pw_alpha))
        return len(node.children) < limit

    for it in range(n_iterations):
        node = root
        path = [node]
        running_cost = scan_cost
        step_merges = [] if return_plan else None

        # ---- Selection ----
        while not _can_widen(node) and node.children:
            total_visits = node.visits
            def ucb(child, pair):
                if child.visits == 0:
                    return float("inf")
                exploit = -child.total_value / child.visits
                explore = c_ucb * math.sqrt(math.log(total_visits + 1) / child.visits)
                return exploit + explore
            pair = max(node.children, key=lambda p: ucb(node.children[p], p))
            a, b = pair
            c = cost_model.join_pair_cost(a, b)
            running_cost += c
            if return_plan:
                step_merges.append((a, b, c))
            node = node.children[pair]
            path.append(node)

        # ---- Expansion ----
        if node.untried:
            pair = rng.choice(node.untried)
            node.untried.remove(pair)
            a, b = pair
            c = cost_model.join_pair_cost(a, b)
            running_cost += c
            if return_plan:
                step_merges.append((a, b, c))
            new_forest = _apply_merge(node.forest, a, b)
            child = _Node(new_forest, _legal_merges(graph, new_forest))
            node.children[pair] = child
            node = child
            path.append(node)

        # ---- Simulation ----
        if return_plan:
            total_cost, rollout_merges = _rollout(
                graph, cost_model, node.forest, running_cost, rng, p_greedy, record=True)
        else:
            total_cost = _rollout(graph, cost_model, node.forest, running_cost, rng, p_greedy)

        if total_cost < best_cost:
            best_cost = total_cost
            if return_plan:
                best_tree = _tree_from_merges(n, step_merges + rollout_merges)

        # ---- Backpropagation ----
        log_cost = math.log(total_cost)
        for nd in path:
            nd.visits += 1
            nd.total_value += log_cost

        if track_curve:
            curve.append((it, best_cost))

    extras = []
    if track_curve:
        extras.append(curve)
    if return_plan:
        extras.append(best_tree)
    if extras:
        return (best_cost, *extras)
    return best_cost


# ---------------------------------------------------------------------------
# 4. Sequential Halving applied to Trees (SHOT, Cazenave & Teytaud 2014)
#
# An alternative way to spend a fixed rollout budget when picking, at each
# decision point, which of several candidate merges to commit to. Instead of
# UCB1's asymmetric "exploit-vs-explore" score, Sequential Halving spends its
# per-decision budget in rounds: evaluate all surviving candidates with an
# equal (small) number of rollouts, discard the worse half, double the
# per-candidate rollout count for the next round, repeat until one survives.
# This is the natural third baseline alongside the two already in
# `bushy_random_restarts`: all three allocate the *same total budget* across
# candidate merge sequences, they just differ in *how* they split it.
# ---------------------------------------------------------------------------
def _sequential_halving_pick(graph, cost_model, forest, pairs, running_cost,
                              budget, rng, p_greedy):
    """Picks one of `pairs` (candidate merges from `forest`) via Sequential
    Halving with a total rollout budget of `budget`, each candidate scored by
    the mean *log*-cost of its allotted rollouts (playing that merge, then a
    p_greedy-biased rollout to a terminal plan). Scoring on log(cost) rather
    than raw cost mirrors MCTS's own backpropagation (Section 4, Formalisation
    du backprop) and matters here for the same reason: costs span many orders
    of magnitude, so ranking by raw arithmetic mean lets a single extreme
    rollout swamp an otherwise-good candidate."""
    survivors = list(pairs)
    if len(survivors) == 1:
        return survivors[0]

    n_rounds = max(1, math.ceil(math.log2(len(survivors))))
    remaining_budget = budget

    for r in range(n_rounds):
        if len(survivors) <= 1:
            break
        rounds_left = n_rounds - r
        per_arm = max(1, remaining_budget // (len(survivors) * rounds_left))
        scored = []
        for (a, b) in survivors:
            c = cost_model.join_pair_cost(a, b)
            child_forest = _apply_merge(forest, a, b)
            total_log = 0.0
            for _ in range(per_arm):
                total_log += math.log(_rollout(
                    graph, cost_model, child_forest, running_cost + c, rng, p_greedy))
            scored.append((total_log / per_arm, (a, b)))
        remaining_budget -= per_arm * len(survivors)
        scored.sort(key=lambda x: x[0])
        keep = max(1, math.ceil(len(scored) / 2))
        survivors = [pair for _, pair in scored[:keep]]

    if len(survivors) == 1:
        return survivors[0]
    # tie-breaker if a round left more than one survivor (budget exhausted):
    # one last single-rollout-per-arm evaluation.
    best_pair, best_cost = None, float("inf")
    for (a, b) in survivors:
        c = cost_model.join_pair_cost(a, b)
        child_forest = _apply_merge(forest, a, b)
        cost = _rollout(graph, cost_model, child_forest, running_cost + c, rng, p_greedy)
        if cost < best_cost:
            best_cost, best_pair = cost, (a, b)
    return best_pair


def bushy_shot(graph, cost_model, total_budget=1500, seed=None, p_greedy=0.75):
    """SHOT: applies Sequential Halving, instead of UCB1, as the tree policy
    at *every* decision point of the plan (root down to the final merge).
    `total_budget` (comparable to MCTS's n_iterations) is split evenly across
    the n-1 decision points, each spent picking that step's merge via
    Sequential Halving over the candidates legal at that point."""
    rng = random.Random(seed)
    n = graph.n
    scan_cost = sum(graph.card[i] for i in range(n))
    forest = _initial_forest(n)
    running_cost = scan_cost

    n_decisions = max(1, n - 1)
    per_decision_budget = max(1, total_budget // n_decisions)

    while len(forest) > 1:
        pairs = _legal_merges(graph, forest)
        if not pairs:
            groups = list(forest)
            pairs = [(groups[0], groups[1])]
        a, b = _sequential_halving_pick(
            graph, cost_model, forest, pairs, running_cost,
            per_decision_budget, rng, p_greedy)
        c = cost_model.join_pair_cost(a, b)
        forest = _apply_merge(forest, a, b)
        running_cost += c

    return running_cost


# ---------------------------------------------------------------------------
# 5. Nested Monte Carlo Search (NMCS, Cazenave 2009)
#
# A tree-free alternative for single-agent sequential construction problems
# (its classical applications are exactly this kind of problem: TSP,
# Morpion Solitaire, SameGame). At level 0, nmcs is just a rollout. At level
# L, it walks forward one merge at a time; at each step it tries *every*
# legal candidate merge, scores each by recursively calling nmcs at level
# L-1 from the resulting state, and commits to whichever candidate scored
# best -- while also remembering the single best complete sequence ever
# encountered anywhere in the recursion (a level-(L-1) call nested deep
# inside a bad branch can still stumble onto a better full plan than the
# one the level-L walk eventually commits to).
# ---------------------------------------------------------------------------
def _nmcs_search(graph, cost_model, forest, running_cost, level, rng, p_greedy):
    if len(forest) <= 1:
        return running_cost, []

    if level == 0:
        return _rollout(graph, cost_model, forest, running_cost, rng, p_greedy, record=True)

    best_cost = float("inf")
    best_merges = None
    cur_forest = forest
    cur_cost = running_cost
    walk_merges = []

    while len(cur_forest) > 1:
        pairs = _legal_merges(graph, cur_forest)
        if not pairs:
            groups = list(cur_forest)
            pairs = [(groups[0], groups[1])]

        best_move = None
        best_local_cost = float("inf")
        for (a, b) in pairs:
            c = cost_model.join_pair_cost(a, b)
            child_forest = _apply_merge(cur_forest, a, b)
            sub_cost, sub_merges = _nmcs_search(
                graph, cost_model, child_forest, cur_cost + c, level - 1, rng, p_greedy)
            if sub_cost < best_local_cost:
                best_local_cost = sub_cost
                best_move = (a, b, c)
            if sub_cost < best_cost:
                best_cost = sub_cost
                best_merges = walk_merges + [(a, b, c)] + sub_merges

        a, b, c = best_move
        cur_forest = _apply_merge(cur_forest, a, b)
        cur_cost += c
        walk_merges.append((a, b, c))

    if cur_cost < best_cost:
        best_cost = cur_cost
        best_merges = walk_merges

    return best_cost, best_merges


def bushy_nmcs(graph, cost_model, level=1, seed=None, p_greedy=0.75, return_plan=False):
    rng = random.Random(seed)
    n = graph.n
    scan_cost = sum(graph.card[i] for i in range(n))
    root_forest = _initial_forest(n)

    best_cost, best_merges = _nmcs_search(
        graph, cost_model, root_forest, scan_cost, level, rng, p_greedy)

    if return_plan:
        return best_cost, _tree_from_merges(n, best_merges)
    return best_cost


def timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - t0
