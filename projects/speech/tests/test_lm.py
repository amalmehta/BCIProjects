import math

import pytest

from speech.lm import BOS, EOS, KneserNeyLM, Trie


def corpus():
    return [s.split() for s in ["the cat sat", "the dog sat", "a cat ran", "the cat ran", "the dog ran"]]


def test_kneser_ney_distributions_sum_to_one():
    vocab = ["the", "a", "cat", "dog", "sat", "ran"]
    lm = KneserNeyLM.train(corpus(), vocab)
    targets = [w for w in lm.vocab if w != BOS]
    for hist in [(), ("the",), ("the", "cat"), ("dog", "ran"), ("never", "seen")]:
        total = sum(math.exp(lm.logprob(hist, w)) for w in targets)
        assert total == pytest.approx(1.0, abs=1e-6), hist


def test_kneser_ney_prefers_seen_continuations():
    lm = KneserNeyLM.train(corpus(), ["the", "a", "cat", "dog", "sat", "ran"])
    assert lm.logprob(("the",), "cat") > lm.logprob(("the",), "ran")
    assert lm.sentence_logprob("the cat sat".split()) > lm.sentence_logprob("sat cat the".split())


def test_trie_shares_nodes_between_homophones():
    lex = {"two": [(31, 34)], "too": [(31, 34)], "to": [(31, 34), (31, 3)]}
    trie = Trie(lex, {"two", "too", "to"})
    node = trie.children[trie.children[0][31]][34]
    assert sorted(trie.words[node]) == ["to", "too", "two"]
    node2 = trie.children[trie.children[0][31]][3]
    assert trie.words[node2] == ["to"]
