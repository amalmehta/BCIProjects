"""Decoder hyper-parameter search: the grid is extended until the optimum is interior."""

from speech.evaluate import search_1d


def test_search_extends_past_edge_until_interior():
    calls = []

    def f(v):
        calls.append(v)
        return (v - 7.0) ** 2

    best, vals, edge = search_1d(f, [0.0, 1.0, 2.0, 3.0], "add", max_expand=10)
    assert best == 7.0 and edge is None
    assert min(vals) < best < max(vals)


def test_search_reports_hard_bound_and_multiplicative_steps():
    best, vals, edge = search_1d(lambda v: v, [0.0, 0.5, 1.0], "mul", lo=0.0)
    assert best == 0.0 and edge == "low"
    best, vals, edge = search_1d(lambda v: (v - 3.0) ** 2, [0.25, 0.5, 1.0], "mul", lo=0.0)
    assert best in (2.0, 4.0) and edge is None and 4.0 in vals
    best, vals, edge = search_1d(lambda v: -v, [1, 2], "add", max_expand=2)
    assert edge == "high" and len(vals) == 4
