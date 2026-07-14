

import math
import random


def geo_mean(xs):
    logs = [math.log(x) for x in xs]
    return math.exp(sum(logs) / len(logs))


def bootstrap_ci(xs, statistic=geo_mean, n_boot=5000, alpha=0.05, seed=0):
    """Percentile bootstrap CI for `statistic` computed over `xs`."""
    rng = random.Random(seed)
    n = len(xs)
    boots = []
    for _ in range(n_boot):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        boots.append(statistic(sample))
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return lo, hi


def paired_sign_flip_test(diffs, n_perm=20000, seed=0):
    """Two-sided permutation (sign-flip) test for H0: median(diffs) == 0.
    Appropriate for paired per-instance comparisons (Greedy vs MCTS on the
    *same* random instance) without assuming normality -- the standard
    non-parametric alternative to a paired t-test when ratios are heavily
    skewed (as they are here, spanning many orders of magnitude)."""
    rng = random.Random(seed)
    observed = sum(diffs)
    n_extreme = 0
    for _ in range(n_perm):
        signed = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(signed) >= abs(observed):
            n_extreme += 1
    return n_extreme / n_perm
