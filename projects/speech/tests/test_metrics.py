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
