"""
cost_model.py

Classic query-optimizer cost model:
  size(S) = product of card(t) for t in S  *  product of selectivity(e) for
            edges e fully inside S
  cost(order) = sum_{k=1..n} size(prefix_k)

This is the standard "sum of intermediate result cardinalities" cost proxy
used to approximate I/O + CPU cost of a left-deep join plan (the same idea
underlying System-R style optimizers and reused in learned-optimizer papers).
Because it only depends on the *set* of tables joined so far (not internal
tree shape), a join order is simply a permutation/sequence of tables, which
is exactly the state representation MCTS operates over.
"""

import math


class CostModel:
    def __init__(self, graph):
        self.g = graph
        self._size_cache = {}

    def set_size(self, table_set):
        """Estimated cardinality of joining this set of tables (order-independent)."""
        key = frozenset(table_set)
        if key in self._size_cache:
            return self._size_cache[key]

        size = 1.0
        for t in table_set:
            size *= self.g.card[t]

        tlist = list(table_set)
        for i in range(len(tlist)):
            for j in range(i + 1, len(tlist)):
                sel = self.g.edge_selectivity(tlist[i], tlist[j])
                if sel is not None:
                    size *= sel

        size = max(size, 1.0)
        self._size_cache[key] = size
        return size

    def incremental_multiplier(self, t, joined_set):
        """Factor by which the joined size grows when adding table t to
        joined_set: card(t) times the selectivity of every edge connecting
        t to a table already in joined_set. O(degree(t)) instead of
        recomputing the whole product from scratch -- this is what makes
        rollouts/greedy tractable on larger graphs (n=40-50+)."""
        mult = float(self.g.card[t])
        for j in self.g.adj[t]:
            if j in joined_set:
                sel = self.g.edge_selectivity(t, j)
                if sel is not None:
                    mult *= sel
        return mult

    def join_pair_cost(self, left_set, right_set):
        """Classic hash-join cost for merging two already-built partial
        results: build a hash table on one side, probe with the other,
        materialize the output. cost = size(L) + size(R) + size(L union R).

        Exception: if this merge produces the COMPLETE join (all n tables),
        we skip adding the output-materialization term. That term is the
        query's final result size, which is fixed and identical no matter
        which plan/algorithm produced it (every valid plan must eventually
        materialize the same final answer) -- including it would let one
        enormous, plan-independent constant swamp the actual differences
        between algorithms (it also causes float64 catastrophic cancellation
        when comparing near-identical huge costs). Excluding it isolates
        exactly the part of the cost that reflects genuine optimization
        quality."""
        left_size = self.set_size(left_set)
        right_size = self.set_size(right_set)
        cost = left_size + right_size
        if len(left_set) + len(right_set) < self.g.n:
            cost += self.set_size(left_set | right_set)
        return cost

    def order_cost(self, order):
        """Total cost of a left-deep plan following `order` (list of table ids)."""
        total = 0.0
        joined = []
        for t in order:
            joined.append(t)
            total += self.set_size(joined)
        return total

    def log_cost(self, order):
        return math.log(self.order_cost(order))
