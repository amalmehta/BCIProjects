"""Lexicon-constrained CTC prefix beam search with a word n-gram LM.

This is a pure-Python stand-in for the WFST decoder of the published systems.
Hypotheses walk a phoneme trie of the vocabulary; a word is committed when the
model emits the word-boundary token ``|`` at a trie node where a word ends
(homophones branch into separate hypotheses). At that point the LM score
``alpha * log P(word | previous two words) + beta`` is added. The acoustic part
is the standard CTC prefix recursion over (blank-ending, non-blank-ending)
probabilities (Graves et al., 2006; Hannun et al., 2014).

``alpha = 0`` gives a "lexicon only" decoder (no language model), which is how
we report WER without an LM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .lm import EOS, KneserNeyLM, Trie
from .phonemes import BLANK_ID, SIL_ID

NEG_INF = -1e30


def _lse(a: float, b: float) -> float:
    if a < b:
        a, b = b, a
    if b <= NEG_INF:
        return a
    return a + math.log1p(math.exp(b - a))


@dataclass
class Hyp:
    pb: float  # log prob of prefix ending in blank
    pnb: float  # log prob of prefix ending in a non-blank token
    lm: float  # accumulated alpha*LM + beta*words
    words: tuple[str, ...]
    node: int  # trie node of the current partial word (0 = between words)
    last: int  # last emitted token id (-1 at start)

    @property
    def ac(self) -> float:
        return _lse(self.pb, self.pnb)

    @property
    def score(self) -> float:
        return self.ac + self.lm


@dataclass
class DecodeResult:
    words: list[str]
    acoustic: float
    lm_score: float  # unweighted n-gram log-probability of the whole sentence (with </s>)
    score: float  # combined score used for ranking


class BeamDecoder:
    """CTC beam search over a lexicon trie with an optional KN n-gram LM."""

    def __init__(self, trie: Trie, lm: KneserNeyLM | None, alpha: float = 0.5, beta: float = 1.0,
                 beam: int = 32, token_prune: float = -8.0, blank_skip: float = 0.999, max_homophones: int = 4):
        self.trie, self.lm = trie, lm
        self.alpha, self.beta, self.beam = alpha, beta, beam
        self.token_prune = token_prune
        self.blank_skip = math.log(blank_skip)
        self.max_homophones = max_homophones

    def _word_lm(self, words: tuple[str, ...], w: str) -> float:
        if self.lm is None or self.alpha == 0:
            return 0.0
        return self.lm.logprob(words, w)

    def decode(self, log_probs: np.ndarray, nbest: int = 1) -> list[DecodeResult]:
        """Decode one utterance's (T, C) CTC log-probabilities into up to ``nbest`` word sequences."""
        children, node_words = self.trie.children, self.trie.words
        hyps: dict[tuple, Hyp] = {((), 0): Hyp(0.0, NEG_INF, 0.0, (), 0, -1)}
        for t in range(log_probs.shape[0]):
            lp = log_probs[t]
            blank_lp = float(lp[BLANK_ID])
            if blank_lp > self.blank_skip:
                cands = []
            else:
                cands = [int(c) for c in np.nonzero(lp > self.token_prune)[0] if c != BLANK_ID]
            new: dict[tuple, Hyp] = {}

            def add(key, words, node, last, lm, pb=NEG_INF, pnb=NEG_INF):
                h = new.get(key)
                if h is None:
                    new[key] = Hyp(pb, pnb, lm, words, node, last)
                else:
                    h.pb = _lse(h.pb, pb)
                    h.pnb = _lse(h.pnb, pnb)

            for key, h in hyps.items():
                total = h.ac
                # stay: blank, or repeat of the last token (collapsed)
                add(key, h.words, h.node, h.last, h.lm, pb=total + blank_lp,
                    pnb=(h.pnb + float(lp[h.last])) if h.last >= 0 else NEG_INF)
                for c in cands:
                    # emitting the same label again needs a blank in between
                    src = h.pb if c == h.last else total
                    if src <= NEG_INF:
                        continue
                    p = src + float(lp[c])
                    if c == SIL_ID:
                        if h.node == 0:  # silence between words / leading silence
                            add((h.words, 0), h.words, 0, SIL_ID, h.lm, pnb=p)
                            continue
                        ws = node_words[h.node]
                        if not ws:
                            continue
                        scored = sorted(((self._word_lm(h.words, w), w) for w in ws), reverse=True)
                        for lm_w, w in scored[: self.max_homophones]:
                            nw = h.words + (w,)
                            add((nw, 0), nw, 0, SIL_ID, h.lm + self.alpha * lm_w + self.beta, pnb=p)
                    else:
                        child = children[h.node].get(c)
                        if child is None:
                            continue
                        add((h.words, child), h.words, child, c, h.lm, pnb=p)
            hyps = dict(sorted(new.items(), key=lambda kv: kv[1].score, reverse=True)[: self.beam])

        # finish: complete a trailing partial word if possible, add end-of-sentence
        finals: list[DecodeResult] = []
        for h in hyps.values():
            cands: list[tuple[tuple[str, ...], float]] = []
            if h.node == 0:
                cands.append((h.words, h.lm))
            else:
                for w in node_words[h.node][: self.max_homophones]:
                    cands.append((h.words + (w,), h.lm + self.alpha * self._word_lm(h.words, w) + self.beta))
            for words, lm in cands:
                eos = self.alpha * self._word_lm(words, EOS)
                full_lm = self.lm.sentence_logprob(list(words)) if self.lm is not None else 0.0
                finals.append(DecodeResult(list(words), h.ac, full_lm, h.ac + lm + eos))
        finals.sort(key=lambda r: r.score, reverse=True)
        seen, out = set(), []
        for r in finals:
            k = tuple(r.words)
            if k not in seen:
                seen.add(k)
                out.append(r)
            if len(out) >= nbest:
                break
        return out or [DecodeResult([], 0.0, 0.0, NEG_INF)]
