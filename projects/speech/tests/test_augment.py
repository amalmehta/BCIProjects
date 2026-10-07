import torch

from speech.augment import AugmentConfig, augment_batch, gauss_smooth, gaussian_kernel


def test_electrode_dropout_zeroes_both_features_of_an_electrode():
    torch.manual_seed(0)
    x = torch.ones(4, 50, 512)
    y, _ = augment_batch(x, torch.full((4,), 50), AugmentConfig(electrode_dropout=0.3))
    tx, sbp = y[..., :256], y[..., 256:]
    assert torch.equal(tx == 0, sbp == 0)
    frac = (tx[:, 0] == 0).float().mean().item()
    assert 0.15 < frac < 0.45


def test_random_cut_shortens_lengths_consistently():
    torch.manual_seed(1)
    x = torch.randn(2, 30, 512)
    L = torch.tensor([30, 20])
    for _ in range(10):
        y, L2 = augment_batch(x, L, AugmentConfig(random_cut=3))
        assert y.shape[1] == 30 - (L - L2)[0]
        assert torch.equal(L - L2, (L - L2)[0].repeat(2))


def test_time_mask_zeroes_a_contiguous_span():
    torch.manual_seed(2)
    x = torch.ones(1, 100, 512)
    y, _ = augment_batch(x, torch.tensor([100]), AugmentConfig(time_mask_num=1, time_mask_max_bins=10))
    zero_rows = (y[0].abs().sum(-1) == 0).nonzero().flatten()
    if len(zero_rows):
        assert zero_rows.max() - zero_rows.min() + 1 == len(zero_rows) <= 10


def test_gaussian_smoothing_preserves_constants():
    k = gaussian_kernel(2.0)
    assert abs(k.sum().item() - 1) < 1e-6
    x = torch.full((1, 40, 3), 2.5)
    y = gauss_smooth(x, k)
    half = len(k) // 2
    assert torch.allclose(y[:, half:-half], x[:, half:-half], atol=1e-5)
