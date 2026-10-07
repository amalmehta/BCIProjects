"""Pronunciation lexicon and a pure-NumPy interpolated Kneser-Ney word trigram LM.

The published systems decode phonemes into words with a Kaldi WFST built from
CMUdict and a large 3/5-gram LM (Willett et al., 2023; Card et al., 2024),
which needs tens to hundreds of GB of RAM. We build a much smaller model of
the same kind that fits on a laptop:

* Corpus: OpenWebText-10k (a public 10k-document sample of OpenWebText, the
  source of some cue sentences), DailyDialog (conversational English, close to
  the Switchboard-style cue sentences), and the cue sentences of the *training*
  trials. Every sentence that also appears in a dev/test/held-out evaluation
  trial is removed to rule out leakage.
* Vocabulary: corpus words with a CMUdict pronunciation and count >= ``min_count``.
* LM: interpolated Kneser-Ney trigram (Kneser & Ney, 1995; Chen & Goodman,
  1999) with absolute discount ``D`` per order, stored as sorted int64 key
  arrays and queried with binary search.
"""

from __future__ import annotations

import io
import json
import lzma
import math
import re
import tarfile
import zipfile
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np

from .metrics import normalize_text
from .phonemes import PHONE_TO_ID, strip_stress

BOS, EOS, UNK = "<s>", "</s>", "<unk>"


# --------------------------------------------------------------------------- lexicon
def load_cmudict(path: Path) -> dict[str, list[tuple[int, ...]]]:
    """Word -> list of pronunciations (tuples of phoneme class ids, stress removed)."""
    lex: dict[str, list[tuple[int, ...]]] = {}
    for line in open(path, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        word, *phones = line.split()
        word = re.sub(r"\(\d+\)$", "", word)
        try:
            pron = tuple(PHONE_TO_ID[strip_stress(p)] for p in phones)
        except KeyError:
            continue
        prons = lex.setdefault(word, [])
        if pron not in prons:
            prons.append(pron)
    return lex


class Trie:
    """Phoneme-prefix trie over a lexicon; node 0 is the root.

    ``children[node]`` maps a phoneme id to a child node and ``words[node]``
    lists the words whose pronunciation ends at that node (homophones share a node).
    """

    def __init__(self, lexicon: dict[str, list[tuple[int, ...]]], vocab: set[str]):
        self.children: list[dict[int, int]] = [{}]
        self.words: list[list[str]] = [[]]
        for w in sorted(vocab):
            for pron in lexicon.get(w, []):
                node = 0
                for p in pron:
                    nxt = self.children[node].get(p)
                    if nxt is None:
                        nxt = len(self.children)
                        self.children[node][p] = nxt
                        self.children.append({})
                        self.words.append([])
                    node = nxt
                if w not in self.words[node]:
                    self.words[node].append(w)


# --------------------------------------------------------------------------- corpus
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _sentences(text: str):
    for s in _SENT_SPLIT.split(text.replace("\n", " ")):
        if re.search(r"\d", s):  # numbers are spoken as words in the cues; skip digit strings
            continue
        s = normalize_text(s)
        n = len(s.split())
        if 2 <= n <= 40:
            yield s


def iter_corpus(lm_dir: Path, max_owt_docs: int | None = None):
    """Yield normalised sentences from OpenWebText-10k and DailyDialog (train)."""
    with lzma.open(lm_dir / "openwebtext-10k" / "10k.xz") as xz, tarfile.open(fileobj=xz, mode="r|") as tar:
        n = 0
        for m in tar:
            if not m.isfile():
                continue
            yield from _sentences(tar.extractfile(m).read().decode("utf-8", "ignore"))
            n += 1
            if max_owt_docs and n >= max_owt_docs:
                break
    with zipfile.ZipFile(lm_dir / "dd_train.zip") as z:
        text = z.read("train/dialogues_train.txt").decode("utf-8", "ignore")
    for line in text.splitlines():
        for utt in line.split("__eou__"):
            utt = re.sub(r"\s+([.,!?;:'])", r"\1", utt).replace(" ' ", "'").replace(" n't", "n't")
            yield from _sentences(utt)


# --------------------------------------------------------------------------- KN trigram
class KneserNeyLM:
    """Interpolated Kneser-Ney trigram LM over integer word ids (natural-log probabilities)."""

    def __init__(self, vocab: list[str], arrays: dict[str, np.ndarray], discount: float = 0.75):
        self.vocab = vocab
        self.wid = {w: i for i, w in enumerate(vocab)}
        self.V = len(vocab)
        self.D = discount
        self.a = arrays
        self._shift = 21  # 2**21 > vocabulary size
        self._uni_total = float(arrays["uni_cc"].sum())

    # keys: (u, v, w) -> (u << 42) | (v << 21) | w
    def _k2(self, u, v):
        return (u << self._shift) | v

    def _k3(self, u, v, w):
        return (u << (2 * self._shift)) | (v << self._shift) | w

    @staticmethod
    def _get(keys: np.ndarray, vals: np.ndarray, k: int) -> float:
        i = np.searchsorted(keys, k)
        return float(vals[i]) if i < len(keys) and keys[i] == k else 0.0

    @classmethod
    def train(cls, sentences, vocab: list[str], discount: float = 0.75) -> "KneserNeyLM":
        """Count n-grams of ``sentences`` (lists of words) mapped onto ``vocab`` (+ <s>, </s>, <unk>)."""
        full = [BOS, EOS, UNK] + [w for w in vocab if w not in (BOS, EOS, UNK)]
        wid = {w: i for i, w in enumerate(full)}
        ids = []
        for s in sentences:
            ids.extend([0, 0] + [wid.get(w, 2) for w in s] + [1])
            ids.append(-1)  # sentence separator
        a = np.asarray(ids, np.int64)
        sh = 21
        # trigram positions: a[i], a[i+1], a[i+2] with no separator and target not BOS
        u, v, w = a[:-2], a[1:-1], a[2:]
        ok = (u >= 0) & (v >= 0) & (w >= 1)
        tri = (u[ok] << 2 * sh) | (v[ok] << sh) | w[ok]
        tri_k, tri_c = np.unique(tri, return_counts=True)
        tu, tv, tw = tri_k >> 2 * sh, (tri_k >> sh) & ((1 << sh) - 1), tri_k & ((1 << sh) - 1)
        # trigram-context statistics: c(u v .) and N1+(u v .)
        ctx = (tu << sh) | tv
        ctx_k, inv = np.unique(ctx, return_inverse=True)
        ctx_c = np.bincount(inv, weights=tri_c).astype(np.int64)
        ctx_n = np.bincount(inv).astype(np.int64)
        # bigram continuation counts N1+(. v w) = number of distinct u preceding (v, w)
        bi = (tv << sh) | tw
        bi_k, bi_inv = np.unique(bi, return_inverse=True)
        bi_cc = np.bincount(bi_inv).astype(np.int64)
        bv = bi_k >> sh
        # bigram-context stats for continuation counts: sum_w N1+(. v w) and N1+(v .)
        bctx_k, binv = np.unique(bv, return_inverse=True)
        bctx_c = np.bincount(binv, weights=bi_cc).astype(np.int64)
        bctx_n = np.bincount(binv).astype(np.int64)
        # unigram continuation counts N1+(. w) = distinct v preceding w
        uw = bi_k & ((1 << sh) - 1)
        uni_cc = np.bincount(uw, minlength=len(full)).astype(np.float64)
        arrays = dict(
            tri_k=tri_k, tri_c=tri_c, ctx_k=ctx_k, ctx_c=ctx_c, ctx_n=ctx_n,
            bi_k=bi_k, bi_cc=bi_cc, bctx_k=bctx_k, bctx_c=bctx_c, bctx_n=bctx_n,
            uni_cc=uni_cc,
        )
        return cls(full, arrays, discount)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, vocab=np.array(self.vocab), discount=self.D, **self.a)

    @classmethod
    def load(cls, path: Path) -> "KneserNeyLM":
        z = np.load(path)
        arrays = {k: z[k] for k in z.files if k not in ("vocab", "discount")}
        return cls([str(w) for w in z["vocab"]], arrays, float(z["discount"]))

    @lru_cache(maxsize=2_000_000)
    def logprob_ids(self, u: int, v: int, w: int) -> float:
        """Natural-log P(w | u, v) under interpolated Kneser-Ney."""
        a, D = self.a, self.D
        # unigram continuation probability, interpolated with uniform for unseen words
        p1 = (a["uni_cc"][w] + 1.0) / (self._uni_total + self.V - 1)  # add-one over all targets except <s>
        # bigram level (continuation counts)
        cb = self._get(a["bctx_k"], a["bctx_c"], v)
        if cb > 0:
            nb = self._get(a["bctx_k"], a["bctx_n"], v)
            ccw = self._get(a["bi_k"], a["bi_cc"], self._k2(v, w))
            p2 = max(ccw - D, 0.0) / cb + D * nb / cb * p1
        else:
            p2 = p1
        # trigram level (raw counts)
        c3 = self._get(a["ctx_k"], a["ctx_c"], self._k2(u, v))
        if c3 > 0:
            n3 = self._get(a["ctx_k"], a["ctx_n"], self._k2(u, v))
            cw = self._get(a["tri_k"], a["tri_c"], self._k3(u, v, w))
            p3 = max(cw - D, 0.0) / c3 + D * n3 / c3 * p2
        else:
            p3 = p2
        return math.log(p3)

    def logprob(self, history: tuple[str, ...], word: str) -> float:
        """log P(word | last two words of history); history excludes <s> padding."""
        h = (BOS, BOS) + tuple(history[-2:])
        u, v = (self.wid.get(x, 2) for x in h[-2:])
        return self.logprob_ids(u, v, self.wid.get(word, 2))

    def sentence_logprob(self, words_: list[str]) -> float:
        """Total log-probability of a sentence including the end-of-sentence token."""
        lp, hist = 0.0, ()
        for w in list(words_) + [EOS]:
            lp += self.logprob(hist, w)
            hist = hist + (w,)
        return lp


# --------------------------------------------------------------------------- build
def build_language_model(lm_dir: Path, train_sentences: list[str], exclude_sentences: list[str],
                         min_count: int = 2, max_owt_docs: int | None = None, out_dir: Path | None = None):
    """Build vocabulary, lexicon trie inputs and KN trigram; returns (lm, vocab, stats)."""
    lexicon = load_cmudict(lm_dir / "cmudict.dict")
    excluded = {normalize_text(s) for s in exclude_sentences}
    corpus, removed = [], 0
    for s in iter_corpus(lm_dir, max_owt_docs):
        if s in excluded:
            removed += 1
            continue
        corpus.append(s.split())
    task = [normalize_text(s).split() for s in train_sentences if normalize_text(s) not in excluded]
    counts = Counter(w for s in corpus for w in s)
    task_words = {w for s in task for w in s}
    vocab = sorted(w for w in set(counts) | task_words
                   if w in lexicon and (counts[w] >= min_count or w in task_words))
    lm = KneserNeyLM.train(corpus + task, vocab)
    eval_words = [w for s in exclude_sentences for w in normalize_text(s).split()]
    vs = set(vocab)
    stats = {
        "corpus_sentences": len(corpus),
        "corpus_words": int(sum(len(s) for s in corpus)),
        "task_train_sentences": len(task),
        "removed_eval_overlap": removed,
        "vocab_size": len(vocab),
        "eval_oov_rate": float(np.mean([w not in vs for w in eval_words])) if eval_words else 0.0,
        "trigrams": int(len(lm.a["tri_k"])),
    }
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        lm.save(out_dir / "kn3.npz")
        (out_dir / "vocab.txt").write_text("\n".join(vocab))
        (out_dir / "lm_stats.json").write_text(json.dumps(stats, indent=2))
    return lm, vocab, stats
