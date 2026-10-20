"""Training, unsupervised cross-day alignment and few-shot recalibration of neural decoders."""
from __future__ import annotations

import copy
import dataclasses
import time

import numpy as np
import torch
import torch.nn.functional as F

from .features import DayData, channel_stats
from .linear import Decoder
from .metrics import r2_score
from .models import GRUDecoder, StableLatentDecoder


def get_device() -> torch.device:
    """CUDA when available (GPU config), else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------------------------
# Decoder wrapper
# ----------------------------------------------------------------------------------------
class TorchDecoder(Decoder):
    """Adapts a torch model to the :class:`~telepathy.linear.Decoder` interface.

    ``day`` selects the read-in slot of a :class:`StableLatentDecoder` (ignored by the GRU).
    """

    def __init__(self, model: torch.nn.Module, day: int, mu: np.ndarray, sd: np.ndarray, vscale: float):
        self.model, self.day, self.mu, self.sd, self.vscale = model.eval(), day, mu, sd, vscale
        self.is_sld = isinstance(model, StableLatentDecoder)

    @torch.no_grad()
    def predict(self, X: np.ndarray, chunk: int = 512) -> np.ndarray:
        dev = next(self.model.parameters()).device
        Xn = torch.as_tensor(self.normalize(X), dtype=torch.float32, device=dev)
        if not self.is_sld:
            y, _ = self.model(Xn[None])
            return (y[0].cpu().numpy() * self.vscale).astype(np.float32)
        if self.model.backbone == "gru":
            y, _, _ = self.model(Xn[None], torch.tensor([self.day], device=dev))
            return (y[0].cpu().numpy() * self.vscale).astype(np.float32)
        ctx = self.model.receptive_field
        out = []
        day = torch.tensor([self.day], device=dev)
        for s in range(0, len(Xn), chunk):
            a = max(0, s - ctx)
            y, _, _ = self.model(Xn[a : s + chunk][None], day)
            out.append(y[0, s - a :])
        return (torch.cat(out).cpu().numpy() * self.vscale).astype(np.float32)

    def reset(self) -> None:
        self._h = None
        self._buf: list[np.ndarray] = []

    @torch.no_grad()
    def step(self, x: np.ndarray) -> np.ndarray:
        xn = torch.as_tensor(self.normalize(x), dtype=torch.float32)[None, None]
        if not self.is_sld:
            y, self._h = self.model(xn, None, self._h)
            return y[0, -1].numpy() * self.vscale
        if self.model.backbone == "gru":
            h, self._h = self.model.latents(xn, torch.tensor([self.day]), self._h, return_state=True)
            return self.model.vel_head(h)[0, -1].numpy() * self.vscale
        self._buf.append(xn[0, 0])
        self._buf = self._buf[-self.model.receptive_field :]
        y, _, _ = self.model(torch.stack(self._buf)[None], torch.tensor([self.day]))
        return y[0, -1].numpy() * self.vscale

    def reset_batch(self, B: int) -> None:
        self._bh = None
        self._bbuf = []
        self._bday = torch.full((B,), self.day, dtype=torch.long)

    @torch.no_grad()
    def step_batch(self, X: np.ndarray) -> np.ndarray:
        """B streams at once: (B, C) counts -> (B, 2) velocity (same maths as :meth:`step`)."""
        xn = torch.as_tensor(self.normalize(X), dtype=torch.float32)[:, None]
        if not self.is_sld:
            y, self._bh = self.model(xn, None, self._bh)
            return (y[:, -1].numpy() * self.vscale).astype(np.float32)
        if self.model.backbone == "gru":
            h, self._bh = self.model.latents(xn, self._bday, self._bh, return_state=True)
            return (self.model.vel_head(h)[:, -1].numpy() * self.vscale).astype(np.float32)
        self._bbuf.append(xn[:, 0])
        self._bbuf = self._bbuf[-self.model.receptive_field :]
        y, _, _ = self.model(torch.stack(self._bbuf, 1), self._bday)
        return (y[:, -1].numpy() * self.vscale).astype(np.float32)


# ----------------------------------------------------------------------------------------
# Training on calibration ("anchor") days
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class TrainConfig:
    """Hyper-parameters for supervised (+ self-supervised) training."""

    steps: int = 2000
    batch: int = 32
    window: int = 64
    warmup_bins: int = 24  # first bins of each window are context only (no loss)
    lr: float = 2e-3
    weight_decay: float = 1e-2
    eval_every: int = 250
    ssl_weight: float = 0.5  # masked-channel Poisson reconstruction (SLD only)
    mask_frac: float = 0.2  # channels masked, on half of each batch, for the SSL loss
    adapter_l2: float = 1e-2  # penalty on per-day adapters' deviation from identity
    augment: bool = True  # channel silencing + gain/offset jitter
    max_drop: float = 0.15
    gain_jitter: float = 0.1
    offset_jitter: float = 0.1
    seed: int = 0


def _prep_days(days: list[DayData]) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], float]:
    """Per-day z-scored inputs (day's own train-block stats) and velocity scale."""
    vscale = float(np.concatenate([d.Y[d.train] for d in days]).std())
    out = []
    for d in days:
        mu, sd = channel_stats(d.X[d.train])
        out.append((mu, sd, ((d.X - mu) / sd).astype(np.float32), (d.Y / vscale).astype(np.float32)))
    return out, vscale


def augment_batch(x: torch.Tensor, mu_over_sd: torch.Tensor, cfg: TrainConfig, g: torch.Generator) -> torch.Tensor:
    """Sussillo-style input perturbations on normalised counts (B, T, C).

    * silence a random fraction (0..max_drop) of channels (count -> 0, i.e. value -mu/sd)
    * per-channel multiplicative gain and additive offset jitter (slow firing-rate drift)
    """
    B, T, C = x.shape
    gain = 1 + cfg.gain_jitter * torch.randn(B, 1, C, generator=g)
    off = cfg.offset_jitter * torch.randn(B, 1, C, generator=g)
    x = x * gain + off
    p = torch.rand(B, 1, 1, generator=g) * cfg.max_drop
    dead = torch.rand(B, 1, C, generator=g) < p
    return torch.where(dead, -mu_over_sd[:, None, :].expand(B, 1, C), x)


def masked_poisson_nll(logr: torch.Tensor, counts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Poisson NLL of ``counts`` under log-rates, averaged over masked entries."""
    nll = F.poisson_nll_loss(logr, counts, log_input=True, full=False, reduction="none")
    return (nll * mask).sum() / mask.sum().clamp(min=1)


def train_neural(kind: str, days: list[DayData], cfg: TrainConfig, model_kwargs: dict | None = None,
                 log=print) -> tuple[torch.nn.Module, float, list[tuple[np.ndarray, np.ndarray]], dict]:
    """Train a GRU ("gru") or SLD ("sld") on the train blocks of ``days``.

    Returns (model, vscale, per-day (mu, sd), history). Model selection by mean validation
    R^2 across days.
    """
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    prep, vscale = _prep_days(days)
    C = days[0].X.shape[1]
    mk = model_kwargs or {}
    model = StableLatentDecoder(C, len(days), **mk) if kind == "sld" else GRUDecoder(C, **mk)
    dev = get_device()
    model.to(dev)
    use_amp = dev.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, cfg.lr, total_steps=cfg.steps, pct_start=0.1)
    counts = [torch.as_tensor(d.X) for d in days]
    best, best_state, hist = -np.inf, None, {"step": [], "val_r2": [], "loss": []}
    t0 = time.time()
    for step in range(1, cfg.steps + 1):
        model.train()
        di = rng.integers(0, len(days), cfg.batch)
        xs, ys, cs = [], [], []
        for k in di:
            tr = days[k].train
            s = rng.integers(tr.start, tr.stop - cfg.window)
            xs.append(prep[k][2][s : s + cfg.window])
            ys.append(prep[k][3][s : s + cfg.window])
            cs.append(counts[k][s : s + cfg.window])
        x = torch.as_tensor(np.stack(xs))
        y = torch.as_tensor(np.stack(ys))
        c = torch.stack(cs)
        day = torch.as_tensor(di)
        mask = torch.zeros(x.shape[0], 1, x.shape[2])
        if cfg.augment:
            mos = torch.as_tensor(np.stack([prep[k][0] / prep[k][1] for k in di]))
            x = augment_batch(x, mos, cfg, g)
        if kind == "sld" and cfg.ssl_weight > 0:  # mask channels on the first half of the batch only
            half = x.shape[0] // 2
            mask[:half] = (torch.rand(half, 1, x.shape[2], generator=g) < cfg.mask_frac).float()
        x, y, c, day, mask = (t.to(dev) for t in (x, y, c, day, mask))
        w = cfg.warmup_bins
        with torch.autocast(device_type=dev.type, enabled=use_amp):
            if kind == "sld":
                vel, logr, _ = model(x * (1 - mask), day)
                loss = F.mse_loss(vel[:, w:].float(), y[:, w:])
                if cfg.ssl_weight > 0:
                    loss = loss + cfg.ssl_weight * masked_poisson_nll(logr[:, w:].float(), c[:, w:],
                                                                      mask.expand_as(c)[:, w:])
                if model.use_adapters and cfg.adapter_l2 > 0:
                    loss = loss + cfg.adapter_l2 * model.adapter_penalty(day)
            else:
                vel, _ = model(x)
                loss = F.mse_loss(vel[:, w:].float(), y[:, w:])
        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        if step % cfg.eval_every == 0 or step == cfg.steps:
            r2 = np.mean([val_r2(model, d, i, prep[i][0], prep[i][1], vscale) for i, d in enumerate(days)])
            hist["step"].append(step)
            hist["val_r2"].append(float(r2))
            hist["loss"].append(float(loss))
            if r2 > best:
                best, best_state = r2, copy.deepcopy(model.state_dict())
            log(f"[{kind}] step {step} loss {loss.item():.4f} val R2 {r2:.4f} ({time.time() - t0:.0f}s)")
    model.load_state_dict(best_state)
    model.cpu().eval()  # evaluation, adaptation and the simulator run on CPU
    return model, vscale, [(p[0], p[1]) for p in prep], hist


def val_r2(model, d: DayData, day: int, mu, sd, vscale: float, ctx: int = 150) -> float:
    """Validation R^2 decoding only the validation block (plus ``ctx`` bins of history)."""
    a = max(0, d.val.start - ctx)
    pred = wrap(model, day, mu, sd, vscale).predict(d.X[a : d.val.stop])[d.val.start - a :]
    return r2_score(d.Y[d.val], pred)


def wrap(model: torch.nn.Module, day: int, mu: np.ndarray, sd: np.ndarray, vscale: float) -> TorchDecoder:
    return TorchDecoder(model, day, mu, sd, vscale)


# ----------------------------------------------------------------------------------------
# Cross-day adaptation of the SLD read-in
# ----------------------------------------------------------------------------------------
@dataclasses.dataclass
class AdaptConfig:
    """Hyper-parameters for unsupervised alignment and few-shot read-in fine-tuning."""

    steps: int = 400
    batch: int = 32
    window: int = 64
    warmup_bins: int = 24
    lr: float = 3e-3
    kl_weight: float = 0.1
    nll_weight: float = 1.0
    adapter_l2: float = 1e-2
    mask_frac: float = 0.2
    fewshot_steps: int = 200
    seed: int = 0


@torch.no_grad()
def latent_stats(model: StableLatentDecoder, days: list[DayData], stats, day_ids: list[int],
                 max_bins: int = 4000) -> tuple[torch.Tensor, torch.Tensor]:
    """Mean and covariance of backbone latents over the train blocks of reference days."""
    hs = []
    for d, (mu, sd), k in zip(days, stats, day_ids):
        x = torch.as_tensor(((d.X[d.train][:max_bins] - mu) / sd).astype(np.float32))
        hs.append(model.latents(x[None], torch.tensor([k]))[0, model.receptive_field :])
    h = torch.cat(hs)
    return h.mean(0), torch.cov(h.T)


def gaussian_kl(mu_q: torch.Tensor, cov_q: torch.Tensor, mu_p: torch.Tensor, cov_p: torch.Tensor,
                eps: float = 1e-3) -> torch.Tensor:
    """KL( N(mu_q, cov_q) || N(mu_p, cov_p) )."""
    D = mu_q.shape[0]
    I = torch.eye(D)
    cq, cp = cov_q + eps * I, cov_p + eps * I
    cp_inv = torch.linalg.inv(cp)
    diff = (mu_p - mu_q)[:, None]
    return 0.5 * (torch.trace(cp_inv @ cq) + (diff.T @ cp_inv @ diff).squeeze() - D
                  + torch.logdet(cp) - torch.logdet(cq))


def adapt_sld(model: StableLatentDecoder, ref: tuple[torch.Tensor, torch.Tensor], X_unlab: np.ndarray,
              mu: np.ndarray, sd: np.ndarray, cfg: AdaptConfig,
              labelled: tuple[np.ndarray, np.ndarray] | None = None, vscale: float = 1.0,
              unsupervised: bool = True, day: int | None = None) -> tuple[StableLatentDecoder, int]:
    """Fit a new day's channel adapter (and read-out gain/offset) with everything else frozen.

    Losses (all optional):
      * unsupervised: masked-channel Poisson NLL of the new day's spikes through the frozen
        backbone, plus ``kl_weight`` x KL between the new day's latent Gaussian and the
        reference (anchor-day) latent Gaussian - a NoMAD-style alignment objective;
      * few-shot: MSE between decoded and true velocity on the labelled bins;
      * always: ``adapter_l2`` x squared deviation of the adapter from the identity.
    If ``day`` is None a new day slot is appended (identity adapter); otherwise training
    continues on that existing slot (e.g. few-shot fine-tuning after unsupervised alignment).
    Returns a *copy* of the model and the index of the adapted slot.
    """
    torch.manual_seed(cfg.seed)
    g = torch.Generator().manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    m = copy.deepcopy(model)
    k = m.add_day() if day is None else day
    for n, p in m.named_parameters():
        p.requires_grad = n in m.DAY_PARAMS
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg.lr)
    # Keep dropout off in the frozen backbone so the KL statistics are deterministic.
    m.eval()
    Xn = ((X_unlab - mu) / sd).astype(np.float32)
    C = torch.as_tensor(X_unlab.astype(np.float32))
    if labelled is not None:
        Xl = ((labelled[0] - mu) / sd).astype(np.float32)
        Yl = (labelled[1] / vscale).astype(np.float32)
    mu_ref, cov_ref = ref
    steps = cfg.steps if labelled is None else cfg.fewshot_steps
    w = cfg.warmup_bins
    for _ in range(steps):
        loss = cfg.adapter_l2 * m.adapter_penalty(torch.tensor([k]))
        if unsupervised and len(Xn) > cfg.window:
            s = rng.integers(0, len(Xn) - cfg.window, cfg.batch)
            x = torch.as_tensor(np.stack([Xn[i : i + cfg.window] for i in s]))
            c = torch.stack([C[i : i + cfg.window] for i in s])
            mask = (torch.rand(x.shape[0], 1, x.shape[2], generator=g) < cfg.mask_frac).float()
            day = torch.full((len(s),), k)
            _, logr, h = m(x * (1 - mask), day)
            if cfg.nll_weight > 0:
                loss = loss + cfg.nll_weight * masked_poisson_nll(logr[:, w:], c[:, w:], mask.expand_as(c)[:, w:])
            if cfg.kl_weight > 0:
                hh = h[:, w:].reshape(-1, h.shape[-1])
                loss = loss + cfg.kl_weight * gaussian_kl(hh.mean(0), torch.cov(hh.T), mu_ref, cov_ref)
        if labelled is not None:
            xl = torch.as_tensor(Xl)[None]
            vel, _, _ = m(xl, torch.tensor([k]))
            loss = loss + F.mse_loss(vel[0, min(w, len(Xl) // 4):], torch.as_tensor(Yl)[min(w, len(Xl) // 4):])
        opt.zero_grad()
        loss.backward()
        opt.step()
    for p in m.parameters():
        p.requires_grad = True
    return m.eval(), k


def finetune_gru(model: GRUDecoder, X: np.ndarray, Y: np.ndarray, mu: np.ndarray, sd: np.ndarray,
                 vscale: float, steps: int = 100, lr: float = 3e-4, seed: int = 0) -> GRUDecoder:
    """Few-shot fine-tuning of all GRU parameters on a short labelled stretch."""
    torch.manual_seed(seed)
    m = copy.deepcopy(model).train()
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    x = torch.as_tensor(((X - mu) / sd).astype(np.float32))[None]
    y = torch.as_tensor((Y / vscale).astype(np.float32))[None]
    w = min(24, x.shape[1] // 4)
    for _ in range(steps):
        out, _ = m(x)
        loss = F.mse_loss(out[:, w:], y[:, w:])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return m.eval()
