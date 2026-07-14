

import random
import itertools


class JoinGraph:
    def __init__(self, n_tables, seed=None, extra_edge_prob=0.15,
                 card_range=(1_000, 2_000_000), sel_range=(0.001, 0.3)):
        self.rng = random.Random(seed)
        self.n = n_tables
        self.card = [self._rand_card(card_range) for _ in range(n_tables)]

        # Build a random spanning tree first (guarantees connectivity),
        # then sprinkle a few extra edges for more complex graphs
        # (star / chain / snowflake-like schemas all arise naturally this way).
        nodes = list(range(n_tables))
        self.rng.shuffle(nodes)
        edges = {}
        for i in range(1, n_tables):
            u = nodes[i]
            v = nodes[self.rng.randint(0, i - 1)]
            edges[self._key(u, v)] = self._rand_sel(sel_range)

        # extra edges (denser query graphs -> more interesting join orders)
        for u, v in itertools.combinations(range(n_tables), 2):
            k = self._key(u, v)
            if k in edges:
                continue
            if self.rng.random() < extra_edge_prob:
                edges[k] = self._rand_sel(sel_range)

        self.edges = edges  # dict {(min(u,v), max(u,v)): selectivity}
        self.adj = {i: set() for i in range(n_tables)}
        for (u, v) in edges:
            self.adj[u].add(v)
            self.adj[v].add(u)

    def _rand_card(self, rng_range):
        lo, hi = rng_range
        # log-uniform: real table sizes span orders of magnitude
        import math
        u = self.rng.random()
        return int(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))))

    def _rand_sel(self, rng_range):
        lo, hi = rng_range
        return self.rng.uniform(lo, hi)

    @staticmethod
    def _key(u, v):
        return (u, v) if u < v else (v, u)

    def edge_selectivity(self, u, v):
        return self.edges.get(self._key(u, v))

    def neighbors_outside(self, joined_set):
        """Tables not yet joined that are adjacent to at least one joined table."""
        result = set()
        for t in joined_set:
            result |= self.adj[t]
        return result - joined_set

    def is_connected_addition(self, joined_set, candidate):
        if not joined_set:
            return True  # starting table, always legal
        return len(self.adj[candidate] & joined_set) > 0
