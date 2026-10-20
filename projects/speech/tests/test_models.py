import pytest
import torch

from speech.models import ConformerDecoder, GRUDecoder, out_lengths, patch
from speech.phonemes import N_CLASSES

SMALL_GRU = dict(hidden=32, layers=2, patch_size=5, patch_stride=2, smooth_std=0.67)
SMALL_CONF = dict(d_model=32, layers=3, heads=4, patch_size=5, patch_stride=2, bottleneck=16,
                  inter_ctc_layers=(1, 2), smooth_std=0.67, dropout=0.0, input_dropout=0.0)


def test_patch_shapes_and_lengths():
    x = torch.arange(2 * 11 * 3, dtype=torch.float32).reshape(2, 11, 3)
    p = patch(x, 5, 2)
    assert p.shape == (2, 4, 15)
    # first patch = bins 0..4 concatenated in time order
    assert torch.equal(p[0, 0], x[0, :5].reshape(-1))
    assert out_lengths(torch.tensor([11, 5, 3]), 5, 2).tolist() == [4, 1, 1]


@pytest.mark.parametrize("model", [
    GRUDecoder(3, **SMALL_GRU),
    GRUDecoder(3, bidirectional=True, **SMALL_GRU),
    ConformerDecoder(3, causal=True, **SMALL_CONF),
    ConformerDecoder(3, causal=False, **SMALL_CONF),
])
def test_output_shapes_are_log_probs(model):
    x = torch.randn(2, 41, 512)
    out = model(x, torch.tensor([41, 30]), torch.tensor([0, 2]))
    assert out["log_probs"].shape == (2, 19, N_CLASSES)
    assert out["lengths"].tolist() == [19, 13]
    assert torch.allclose(out["log_probs"].exp().sum(-1), torch.ones(2, 19), atol=1e-5)


def test_conformer_returns_intermediate_ctc_heads():
    m = ConformerDecoder(2, **SMALL_CONF)
    out = m(torch.randn(1, 41, 512), torch.tensor([41]), torch.tensor([0]))
    assert len(out["inter_log_probs"]) == 2


@pytest.mark.parametrize("make", [lambda: GRUDecoder(1, **SMALL_GRU),
                                  lambda: ConformerDecoder(1, causal=True, **SMALL_CONF)])
def test_causal_models_do_not_look_ahead(make):
    """Perturbing input bins after time t only changes outputs whose window (+ smoothing) reaches t."""
    torch.manual_seed(0)
    m = make().eval()
    x = torch.randn(1, 61, 512)
    y = x.clone()
    t = 40
    y[:, t:] += 5 * torch.randn_like(y[:, t:])
    L, d = torch.tensor([61]), torch.tensor([0])
    a, b = m(x, L, d)["log_probs"][0], m(y, L, d)["log_probs"][0]
    half_k = (len(m.kernel) - 1) // 2
    # frame f covers bins [2f, 2f+4]; smoothing reaches half_k bins further
    last_clean = max(f for f in range(a.shape[0]) if 2 * f + 4 + half_k < t)
    assert torch.allclose(a[: last_clean + 1], b[: last_clean + 1], atol=1e-5)
    assert not torch.allclose(a[last_clean + 1 :], b[last_clean + 1 :])


def test_day_layers_start_as_identity_and_are_day_specific():
    m = GRUDecoder(3, **SMALL_GRU)
    assert torch.equal(m.day.weight[1], torch.eye(512))
    shared = GRUDecoder(3, per_day=False, **SMALL_GRU)
    assert shared.day.weight.shape[0] == 1


def test_blank_bias_initialisation_sets_initial_blank_posterior():
    import math

    from speech.models import ConformerDecoder, GRUDecoder

    for cls, kw in ((GRUDecoder, dict(hidden=16, layers=1)), (ConformerDecoder, dict(d_model=16, layers=1, heads=2, bottleneck=8))):
        m = cls(1, patch_size=3, patch_stride=2, blank_bias=4.0, **kw)
        assert m.out.bias[0].item() == 4.0 and m.out.bias[1:].abs().sum().item() == 0.0
        # with the bias alone, p(blank) = e^4 / (e^4 + 40)
        assert math.exp(4) / (math.exp(4) + 40) > 0.5
