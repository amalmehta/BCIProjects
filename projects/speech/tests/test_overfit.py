"""A tiny decoder must be able to memorise a handful of synthetic sentences."""

import torch
import torch.nn.functional as F

from speech.ctc import greedy_decode
from speech.models import GRUDecoder


def test_tiny_gru_overfits_synthetic_sequences():
    torch.manual_seed(0)
    B, T = 4, 48
    # each phoneme class has a fixed random spatial pattern; neural data = pattern + noise
    patterns = torch.randn(41, 512)
    labels = [torch.tensor(s) for s in ([5, 9, 40, 12], [3, 40, 7], [20, 40, 21, 22], [9, 5, 40])]
    x = torch.zeros(B, T, 512)
    for b, lab in enumerate(labels):
        seg = T // (len(lab) + 1)
        for i, c in enumerate(lab):
            x[b, i * seg + 2 : (i + 1) * seg] = patterns[c]
    x += 0.1 * torch.randn_like(x)
    y = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)
    y_len = torch.tensor([len(l) for l in labels])
    torch.manual_seed(0)
    model = GRUDecoder(1, hidden=128, layers=1, patch_size=3, patch_stride=2, dropout=0.0, input_dropout=0.0, smooth_std=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=5e-3)
    day = torch.zeros(B, dtype=torch.long)
    first = None
    for _ in range(200):
        out = model(x, torch.full((B,), T), day)
        loss = F.ctc_loss(out["log_probs"].transpose(0, 1), y, out["lengths"], y_len, zero_infinity=True)
        first = first or loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.02 * first
    model.eval()
    lp = model(x, torch.full((B,), T), day)["log_probs"].detach().numpy()
    assert [greedy_decode(lp[b]) for b in range(B)] == [l.tolist() for l in labels]
