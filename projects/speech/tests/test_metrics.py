import pytest

from speech.metrics import edit_distance, error_rate, normalize_text, wer


def test_edit_distance_hand_computed():
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance([], [1, 2]) == 2
    assert edit_distance([1, 2, 3], [1, 2, 3]) == 0
    assert edit_distance([1, 2, 3], [3]) == 2


def test_corpus_level_error_rate_pools_edits():
    # 1 edit over 4 tokens and 2 edits over 1 token -> 3/5, not mean(0.25, 2.0)
    assert error_rate([[1, 2, 3, 4], [5]], [[1, 2, 3, 0], [6, 7]]) == pytest.approx(3 / 5)


def test_wer_normalises_case_and_punctuation():
    assert normalize_text("Hello, World!  It's ok-ish.") == "hello world it's ok ish"
    assert wer(["How are you?"], ["how are you"]) == 0.0
    assert wer(["a b c d"], ["a x c"]) == pytest.approx(2 / 4)


def test_bootstrap_from_edits_matches_rate_and_brackets_it():
    import numpy as np
    from speech.metrics import bootstrap_ci_edits, sentence_edits

    refs = [[1, 2, 3, 4]] * 50 + [[5, 6]] * 50
    hyps = [[1, 2, 3, 0]] * 50 + [[5, 6]] * 50
    e, t = sentence_edits(refs, hyps)
    assert e.sum() / t.sum() == pytest.approx(error_rate(refs, hyps))
    lo, hi = bootstrap_ci_edits(e, t)
    assert lo <= error_rate(refs, hyps) <= hi


def test_paired_bootstrap_detects_consistent_difference_and_not_a_null_one():
    import numpy as np
    from speech.metrics import paired_bootstrap

    rng = np.random.default_rng(0)
    t = rng.integers(5, 15, 300).astype(float)
    e_b = rng.binomial(t.astype(int), 0.5).astype(float)
    better = paired_bootstrap(np.maximum(e_b - 1, 0), e_b, t)
    assert better["diff"] < 0 and better["p_two_sided"] < 0.01 and better["ci95"][1] < 0
    same = paired_bootstrap(e_b, e_b, t)
    assert same["diff"] == 0 and same["p_two_sided"] == 1.0
