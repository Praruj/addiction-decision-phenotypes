"""A GRU that learns to predict the next card a person will pick.

It is trained on other people only. At test time it reads a new person's own history
(choices and payoffs so far) and must adapt in context, with no per-person fitting.
Its hidden state is also a learned, theory-free summary of how someone plays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from .models import Batch

IN_DIM = 9


def inputs(b: Batch) -> torch.Tensor:
    """Input at step t describes trial t-1 (previous choice and its payoff)."""
    N, T = b.choice.shape
    oh = nn.functional.one_hot(b.choice, 4).float()
    x = b.out.float().unsqueeze(-1)
    feats = torch.cat([oh, x.clamp(min=0), x.clamp(max=0), x, (x >= 0).float()], -1)
    prev = torch.cat([torch.zeros(N, 1, 8), feats[:, :-1]], 1)
    flag = torch.zeros(N, T, 1)
    flag[:, 0] = 1
    return torch.cat([prev, flag], -1)


class ChoiceGRU(nn.Module):
    def __init__(self, hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(IN_DIM, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 4)

    def forward(self, b: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.gru(inputs(b))
        return self.head(self.drop(h)), h


def _nll(logits: torch.Tensor, b: Batch, m: torch.Tensor) -> torch.Tensor:
    lp = torch.log_softmax(logits, -1).gather(-1, b.choice.unsqueeze(-1)).squeeze(-1)
    return -(lp * m).sum() / m.sum()


def _take(b: Batch, idx) -> Batch:
    return Batch(b.choice[idx], b.out[idx], b.mask[idx])


@dataclass
class TrainedGRU:
    nets: list[ChoiceGRU]

    @torch.no_grad()
    def proba(self, b: Batch) -> np.ndarray:
        ps = []
        for n in self.nets:
            n.eval()
            ps.append(torch.softmax(n(b)[0], -1))
        return torch.stack(ps).mean(0).double().numpy()

    @torch.no_grad()
    def embed(self, b: Batch) -> np.ndarray:
        """Mean hidden state over the game, concatenated across ensemble members."""
        out = []
        for n in self.nets:
            n.eval()
            h = n(b)[1]
            m = b.mask.float().unsqueeze(-1)
            out.append(((h * m).sum(1) / m.sum(1)).numpy())
        return np.concatenate(out, 1)


def train(b: Batch, seeds=(0, 1, 2), hidden: int = 32, epochs: int = 600,
          patience: int = 40, lr: float = 1e-2, wd: float = 1e-4,
          val_frac: float = 0.15) -> TrainedGRU:
    """Train a small seed ensemble; early stopping on held-out training people."""
    nets = []
    N = b.choice.shape[0]
    for seed in seeds:
        torch.manual_seed(seed)
        perm = torch.randperm(N)
        nv = max(1, int(round(val_frac * N)))
        tr, va = _take(b, perm[nv:]), _take(b, perm[:nv])
        net = ChoiceGRU(hidden)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=wd)
        best, best_state, bad = np.inf, None, 0
        for _ in range(epochs):
            net.train()
            opt.zero_grad()
            loss = _nll(net(tr)[0], tr, tr.mask.float())
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            net.eval()
            with torch.no_grad():
                v = _nll(net(va)[0], va, va.mask.float()).item()
            if v < best - 1e-4:
                best, bad = v, 0
                best_state = {k: t.clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        net.load_state_dict(best_state)
        nets.append(net)
    return TrainedGRU(nets)
