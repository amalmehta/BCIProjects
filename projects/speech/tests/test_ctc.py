import numpy as np

from speech.ctc import greedy_decode, greedy_with_times


def onehot(seq, C=41):
    lp = np.full((len(seq), C), -20.0)
    lp[np.arange(len(seq)), seq] = 0.0
    return lp


def test_greedy_merges_repeats_and_drops_blanks():
    lp = onehot([0, 5, 5, 0, 5, 7, 7, 0, 40])
    assert greedy_decode(lp) == [5, 5, 7, 40]
    assert greedy_with_times(lp) == [(5, 1), (5, 4), (7, 5), (40, 8)]
