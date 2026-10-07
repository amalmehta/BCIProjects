import math

import numpy as np
import torch
from skimage.metrics import structural_similarity

from blindsight.encoders import (IntensityEncoder, LearnedEncoder, SafetyLayer, bilinear_taps,
                                 gaussian_blur, sample_at)
from blindsight.metrics import psnr, ssim


def test_ssim_identity_and_matches_skimage():
    g = torch.Generator().manual_seed(0)
    x = torch.rand(3, 1, 32, 32, generator=g)
    y = (x + 0.3 * torch.rand(3, 1, 32, 32, generator=g)).clamp(0, 1)
    assert torch.allclose(ssim(x, x), torch.ones(3), atol=1e-5)
    ref = [structural_similarity(x[i, 0].numpy(), y[i, 0].numpy(), data_range=1.0, gaussian_weights=True,
                                 sigma=1.5, use_sample_covariance=False) for i in range(3)]
    np.testing.assert_allclose(ssim(x, y).numpy(), ref, atol=1e-4)


def test_psnr_hand_computed():
    x = torch.zeros(1, 1, 4, 4)
    y = torch.full((1, 1, 4, 4), 0.1)  # MSE = 0.01 -> 20 dB
    assert math.isclose(float(psnr(x, y)), 20.0, rel_tol=1e-5)


def test_bilinear_taps_hit_pixel_centres():
    R, fov = 8, 8.0
    img = torch.arange(R * R, dtype=torch.float32).view(1, 1, R, R)
    # centre of pixel (row 2, col 5): x = 5.5 - 4 = 1.5, y = 4 - 2.5 = 1.5
    idx, w = bilinear_taps(torch.tensor([1.5]), torch.tensor([1.5]), fov, R)
    assert math.isclose(float(sample_at(img, idx, w)), 2 * R + 5, abs_tol=1e-4)
    # halfway between columns 5 and 6 -> mean of the two
    idx, w = bilinear_taps(torch.tensor([2.0]), torch.tensor([1.5]), fov, R)
    assert math.isclose(float(sample_at(img, idx, w)), 2 * R + 5.5, abs_tol=1e-4)


def test_gaussian_blur_preserves_constant():
    x = torch.full((1, 1, 16, 16), 0.7)
    assert torch.allclose(gaussian_blur(x, 1.0), x, atol=1e-6)


def test_safety_layer_clamps_and_limits_active():
    s = SafetyLayer(100.0, 3)
    a = s(torch.tensor([[150.0, -5.0, 20.0, 80.0, 10.0, 90.0]]))
    assert a.max() == 100.0 and a.min() == 0.0 and int((a > 0).sum()) == 3


def test_intensity_encoder_black_image_is_silent(setup):
    enc = IntensityEncoder(setup.context(setup.patient(3)))
    R = setup.params.resolution
    assert enc(torch.zeros(2, 1, R, R)).abs().max() == 0
    a = enc(torch.ones(1, 1, R, R))
    alive = setup.patient(3).alive.bool()
    assert torch.allclose(a[0, alive], torch.full_like(a[0, alive], 100.0))
    assert a[0, ~alive].abs().max() == 0


def test_learned_encoder_shapes_and_range(setup):
    enc = LearnedEncoder(setup.context(calibrated=False), width=8)
    R = setup.params.resolution
    a = enc(torch.rand(3, 1, R, R))
    assert a.shape == (3, setup.n) and a.min() >= 0 and a.max() <= setup.params.max_current_ua
