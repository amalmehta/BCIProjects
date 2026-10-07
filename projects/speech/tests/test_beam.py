import numpy as np

from speech.beam import BeamDecoder
from speech.lm import KneserNeyLM, Trie
from speech.phonemes import PHONE_TO_ID, SIL_ID

LEX = {
    "the": [("DH", "AH")],
    "cat": [("K", "AE", "T")],
    "sat": [("S", "AE", "T")],
    "mat": [("M", "AE", "T")],
    "on": [("AA", "N")],
}


def lexicon():
    return {w: [tuple(PHONE_TO_ID[p] for p in pr) for pr in prons] for w, prons in LEX.items()}


def posteriors(words, noise=0.0, seed=0):
    """Peaky CTC posteriors: each phoneme for 2 frames, a blank, SIL after each word."""
    seq = []
    for w in words:
        for p in LEX[w][0]:
            seq += [PHONE_TO_ID[p]] * 2 + [0]
        seq += [SIL_ID, 0]
    rng = np.random.default_rng(seed)
    lp = rng.normal(0, noise, (len(seq), 41))
    lp[np.arange(len(seq)), seq] += 8
    return lp - np.log(np.exp(lp).sum(1, keepdims=True))


def test_beam_recovers_sentence_from_clean_posteriors():
    trie = Trie(lexicon(), set(LEX))
    dec = BeamDecoder(trie, None, alpha=0.0, beta=0.0, beam=8)
    words = ["the", "cat", "sat", "on", "the", "mat"]
    assert dec.decode(posteriors(words))[0].words == words


def test_lm_resolves_acoustic_ambiguity():
    trie = Trie(lexicon(), set(LEX))
    corpus = [s.split() for s in ["the cat sat on the mat"] * 20 + ["the mat sat"]]
    lm = KneserNeyLM.train(corpus, list(LEX))
    lp = posteriors(["the", "cat", "sat"])
    # make "cat" acoustically ambiguous with "mat": first phoneme K vs M equally likely
    k, m = PHONE_TO_ID["K"], PHONE_TO_ID["M"]
    for t in (8, 9):  # the two frames of "K" in "cat"
        lp[t, k] = lp[t, m] = np.log(0.5)
    no_lm = BeamDecoder(trie, None, alpha=0.0, beta=0.0, beam=8).decode(lp, nbest=2)
    with_lm = BeamDecoder(trie, lm, alpha=1.0, beta=0.0, beam=8).decode(lp)[0]
    assert {tuple(r.words) for r in no_lm} == {("the", "cat", "sat"), ("the", "mat", "sat")}
    assert with_lm.words == ["the", "cat", "sat"]
