"""Reinforcement-learning models of IGT choice, fitted by batched maximum likelihood.

Three standard models from computational psychiatry, written as in hBayesDM
(Ahn, Haines & Zhang, 2017) with outcomes divided by 100:

* PVL-Delta  (Ahn et al., 2008): prospect-theory utility + delta-rule learning.
* VPP        (Worthy et al., 2013): PVL-Delta plus a win/loss-dependent perseverance trace.
* ORL        (Haines et al., 2018): separate learning from rewards and punishments,
  a win-frequency signal and perseverance.

Every subject and every random restart is one row of a tensor, so all of them are fitted
together with Adam and exact autograd gradients. Parameters live on an unconstrained scale
and are squashed into their bounds with a sigmoid; a weak N(0, 2^2) prior on that scale
keeps estimates off the bounds (penalised maximum likelihood).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import N_TRIALS, Subject


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: tuple[str, ...]
    lo: tuple[float, ...]
    hi: tuple[float, ...]

    @property
    def n(self) -> int:
        return len(self.params)


SPECS = {
    "pvl_delta": ModelSpec("pvl_delta", ("A", "alpha", "cons", "lambda"),
                           (0, 0, 0, 0), (1, 2, 5, 10)),
    "vpp": ModelSpec("vpp", ("A", "alpha", "cons", "lambda", "epP", "epN", "K", "w"),
                     (0, 0, 0, 0, -5, -5, 0, 0), (1, 2, 5, 10, 5, 5, 1, 1)),
    "orl": ModelSpec("orl", ("Arew", "Apun", "K", "betaF", "betaP"),
                     (0, 0, 0, -10, -10), (1, 1, 5, 10, 10)),
}


# --------------------------------------------------------------------------- tensors
@dataclass
class Batch:
    choice: torch.Tensor   # (N, T) long
    out: torch.Tensor      # (N, T) net outcome / 100
    mask: torch.Tensor     # (N, T) bool, real trial


def to_batch(subs: list[Subject], T: int = N_TRIALS) -> Batch:
    N = len(subs)
    choice = np.zeros((N, T), int)
    out = np.zeros((N, T))
    mask = np.zeros((N, T), bool)
    for i, s in enumerate(subs):
        t = len(s.choice)
        choice[i, :t], out[i, :t], mask[i, :t] = s.choice, s.net / 100, True
    return Batch(torch.tensor(choice), torch.tensor(out, dtype=torch.float64),
                 torch.tensor(mask))


def constrain(spec: ModelSpec, z: torch.Tensor) -> torch.Tensor:
    lo = torch.tensor(spec.lo, dtype=z.dtype)
    hi = torch.tensor(spec.hi, dtype=z.dtype)
    return lo + (hi - lo) * torch.sigmoid(z)


def _utility(x: torch.Tensor, alpha: torch.Tensor, lam: torch.Tensor) -> torch.Tensor:
    # |x|^alpha, written so that x == 0 has a finite gradient
    ax = torch.where(x == 0, torch.ones_like(x), x.abs())
    mag = torch.where(x == 0, torch.zeros_like(x), ax ** alpha)
    return torch.where(x >= 0, mag, -lam * mag)


# --------------------------------------------------------------------------- models
def logits(model: str, theta: torch.Tensor, b: Batch) -> torch.Tensor:
    """One-step-ahead choice logits (N, T, 4): trial t uses only trials < t."""
    N, T = b.choice.shape
    onehot = torch.nn.functional.one_hot(b.choice, 4).to(theta.dtype)
    out_logits = []
    if model in ("pvl_delta", "vpp"):
        A, alpha, cons, lam = (theta[:, i:i + 1] for i in range(4))
        sens = 3 ** cons - 1
        ev = torch.zeros(N, 4, dtype=theta.dtype)
        if model == "vpp":
            epP, epN, K, w = (theta[:, i:i + 1] for i in range(4, 8))
            pers = torch.zeros(N, 4, dtype=theta.dtype)
        for t in range(T):
            v = ev if model == "pvl_delta" else w * ev + (1 - w) * pers
            out_logits.append(sens * v)
            x = b.out[:, t:t + 1]
            u = _utility(x, alpha, lam)
            oh = onehot[:, t]
            ev = ev + oh * A * (u - ev)
            if model == "vpp":
                pers = pers * K + oh * torch.where(x >= 0, epP, epN)
    elif model == "orl":
        Arew, Apun, K, betaF, betaP = (theta[:, i:i + 1] for i in range(5))
        Ktr = 3 ** K - 1
        ev = torch.zeros(N, 4, dtype=theta.dtype)
        ef = torch.zeros(N, 4, dtype=theta.dtype)
        pers = torch.zeros(N, 4, dtype=theta.dtype)
        for t in range(T):
            out_logits.append(ev + ef * betaF + pers * betaP)
            x = b.out[:, t:t + 1]
            sgn = torch.sign(x)
            oh = onehot[:, t]
            win = x >= 0
            a_ch = torch.where(win, Arew, Apun)      # chosen deck
            a_un = torch.where(win, Apun, Arew)      # fictive update of unchosen decks
            ev = ev + oh * a_ch * (x - ev)
            ef = ef + oh * a_ch * (sgn - ef) + (1 - oh) * a_un * (-sgn / 3 - ef)
            pers = (pers * (1 - oh) + oh) / (1 + Ktr)
    else:
        raise ValueError(model)
    return torch.stack(out_logits, 1)


def trial_nll(model: str, theta: torch.Tensor, b: Batch) -> torch.Tensor:
    """Negative log-likelihood of every trial, (N, T)."""
    lp = torch.log_softmax(logits(model, theta, b), -1)
    return -lp.gather(-1, b.choice.unsqueeze(-1)).squeeze(-1)


# --------------------------------------------------------------------------- fitting
@dataclass
class Fit:
    model: str
    params: np.ndarray   # (N, P) constrained
    z: np.ndarray        # (N, P) unconstrained (used as ML features)
    nll: np.ndarray      # (N,) on the fitted trials
    k: int               # number of fitted trials per subject (array)


def fit(model: str, b: Batch, fit_mask: torch.Tensor | None = None, restarts: int = 12,
        steps: int = 1200, lr: float = 0.05, prior_sd: float = 2.0, seed: int = 0) -> Fit:
    """Fit each subject separately (but in one batch) on the trials in `fit_mask`."""
    spec = SPECS[model]
    N, T = b.choice.shape
    fm = b.mask if fit_mask is None else (fit_mask & b.mask)
    g = torch.Generator().manual_seed(seed)
    R = restarts
    rep = lambda x: x.repeat_interleave(R, 0)
    bb = Batch(rep(b.choice), rep(b.out), rep(b.mask))
    fmr = rep(fm).to(torch.float64)
    z = (torch.randn(N * R, spec.n, generator=g, dtype=torch.float64) * 1.5).requires_grad_()
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        nll = (trial_nll(model, constrain(spec, z), bb) * fmr).sum(1)
        loss = nll + 0.5 * (z / prior_sd).pow(2).sum(1)
        loss.sum().backward()
        opt.step()
        sched.step()
    with torch.no_grad():
        nll = (trial_nll(model, constrain(spec, z), bb) * fmr).sum(1)
        obj = nll + 0.5 * (z / prior_sd).pow(2).sum(1)
        best = obj.view(N, R).argmin(1)
        idx = torch.arange(N) * R + best
        zb = z[idx]
    return Fit(model, constrain(spec, zb).numpy(), zb.numpy(), nll[idx].numpy(),
               fm.sum(1).numpy())


def predict_proba(model: str, params: np.ndarray, b: Batch) -> np.ndarray:
    with torch.no_grad():
        th = torch.tensor(params, dtype=torch.float64)
        return torch.softmax(logits(model, th, b), -1).numpy()


def bic(f: Fit) -> np.ndarray:
    return 2 * f.nll + SPECS[f.model].n * np.log(f.k)


def simulate(model: str, params: np.ndarray, schedule: dict, T: int = N_TRIALS,
             seed: int = 0) -> list[tuple[int, float, float]]:
    """Let a fitted model play the task (posterior-predictive check)."""
    rng = np.random.default_rng(seed)
    draws = [0, 0, 0, 0]
    hist: list[tuple[int, float, float]] = []
    for t in range(T):
        if hist:
            c = torch.tensor([[h[0] for h in hist] + [0]])
            o = torch.tensor([[(h[1] + h[2]) / 100 for h in hist] + [0.0]], dtype=torch.float64)
        else:
            c = torch.zeros(1, 1, dtype=torch.long)
            o = torch.zeros(1, 1, dtype=torch.float64)
        b = Batch(c, o, torch.ones_like(c, dtype=torch.bool))
        p = predict_proba(model, params[None], b)[0, -1]
        d = int(rng.choice(4, p=p / p.sum()))
        gain, loss = draw(schedule, d, draws[d])
        draws[d] += 1
        hist.append((d, gain, loss))
    return hist


def draw(schedule: dict, deck: int, k: int) -> tuple[float, float]:
    """k-th card of a deck; beyond the observed cards the sequence repeats."""
    seq = schedule[deck]
    g, l = seq[k % len(seq)]
    return float(g), float(l)
