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


class _FakeDecoder:
    """Stands in for the beam-search pool: hypothesis quality depends on (alpha, beta, beam)."""

    def __init__(self, best=(1.0, 2.0)):
        self.best = best

    def __call__(self, lps, alpha, beta, nbest=1, beam=None):
        out = []
        for ref in lps:  # here "posteriors" are the reference strings themselves
            w = ref.split()
            import math

            k = int(round(4 * abs(math.log2(max(alpha, 1e-3) / self.best[0])) + abs(beta - self.best[1])))
            hyp = " ".join(w[k:]) if k else ref
            out.append([(hyp, 0.0, 0.0, 0.0), (ref, -1.0, -5.0, -1.0)][:nbest])
        return out


class _FakeScorer:
    def score(self, sents):
        return [float(len(s.split())) for s in sents]


def test_tune_decoder_finds_interior_optimum_on_dev():
    from speech.evaluate import rescore_tune, tune_decoder

    refs = [" ".join("abcdefghijklmnopqrstuvwxyz")] * 5
    # the alpha grid stops below the optimum (1.0): the search must extend it upwards
    ecfg = {"beam": 16, "beams": [8, 16, 32], "alphas": [0.0625, 0.125, 0.25], "betas": [-2.0, 0.0, 2.0, 4.0],
            "betas_lexicon": [-2.0, 0.0, 2.0], "tune_rounds": 2}
    out, log = tune_decoder(_FakeDecoder(), refs, refs, ecfg)
    assert out["alpha"] == 1.0 and out["beta"] == 2.0 and out["dev_wer_ngram"] == 0.0
    assert out["interior"]["alpha"] and out["interior"]["beta"]
    nb = _FakeDecoder((1.0, 0.0))(refs, 1.0, 2.0, nbest=2)  # top hypothesis truncated, 2nd = reference
    g, b, dev_wer, glog = rescore_tune(nb, refs, _FakeScorer(), 1.0, 2.0, [0.0, 0.5, 1.0], [-1.0, 1.0])
    assert dev_wer == 0.0 and g > 0
