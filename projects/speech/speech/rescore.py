"""n-best rescoring with a small causal LLM (GPT-2 small, 124M parameters).

The published systems rescore their WFST n-best lists with OPT-6.7B (Willett et
al., 2023; Card et al., 2024). On a CPU we use GPT-2 small. The final score of a
hypothesis is

    acoustic + alpha * ngram + beta * n_words + gamma * llm,

with alpha and beta fixed from the n-gram decoding stage and gamma tuned on the dev set.
"""

from __future__ import annotations

from pathlib import Path

import torch


class GPT2Scorer:
    """Sentence log-likelihoods under GPT-2 (natural log, summed over tokens)."""

    def __init__(self, model_dir: str | Path, device: str = "cpu"):
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast

        self.tok = GPT2TokenizerFast.from_pretrained(model_dir)
        self.model = GPT2LMHeadModel.from_pretrained(model_dir).to(device).eval()
        self.device = device
        self._cache: dict[str, float] = {}

    @torch.no_grad()
    def score(self, sentences: list[str], batch_size: int = 16) -> list[float]:
        """Return log P(sentence) for each sentence (prefixed by the BOS/EOS token)."""
        todo = [s for s in dict.fromkeys(sentences) if s not in self._cache]
        for i in range(0, len(todo), batch_size):
            chunk = todo[i : i + batch_size]
            texts = [self.tok.bos_token + (s[:1].upper() + s[1:] + "." if s else "") for s in chunk]
            enc = [self.tok(t)["input_ids"] for t in texts]
            L = max(len(e) for e in enc)
            ids = torch.full((len(enc), L), self.tok.eos_token_id)
            mask = torch.zeros((len(enc), L), dtype=torch.long)
            for j, e in enumerate(enc):
                ids[j, : len(e)] = torch.tensor(e)
                mask[j, : len(e)] = 1
            logits = self.model(ids.to(self.device), attention_mask=mask.to(self.device)).logits.float()
            lp = logits[:, :-1].log_softmax(-1).gather(-1, ids[:, 1:, None].to(self.device)).squeeze(-1)
            lp = (lp * mask[:, 1:].to(self.device)).sum(1).cpu().tolist()
            for s, v in zip(chunk, lp):
                self._cache[s] = v
        return [self._cache[s] for s in sentences]
