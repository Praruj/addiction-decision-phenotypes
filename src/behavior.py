"""Model-free behavioural summaries of one IGT game."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import DECKS, Subject


def summary(choice: np.ndarray, net: np.ndarray) -> dict[str, float]:
    c = np.asarray(choice)
    good = np.isin(c, [2, 3]).astype(float)
    f: dict[str, float] = {}
    for b in range(5):
        blk = good[b * 20:(b + 1) * 20]
        f[f"net_block{b + 1}"] = float(2 * blk.mean() - 1) * 20 if len(blk) else np.nan
    f["net_score"] = float(2 * good.mean() - 1) * len(c)
    for d, name in enumerate(DECKS):
        f[f"p_{name}"] = float((c == d).mean())
    stay = c[1:] == c[:-1]
    win = net[:-1] >= 0
    f["win_stay"] = float(stay[win].mean()) if win.any() else np.nan
    f["lose_shift"] = float((~stay[~win]).mean()) if (~win).any() else np.nan
    f["switch_rate"] = float((~stay).mean())
    p = np.bincount(c, minlength=4) / len(c)
    f["entropy"] = float(-(p[p > 0] * np.log2(p[p > 0])).sum())
    f["money"] = float(net.sum())
    return f


def table(subs: list[Subject]) -> pd.DataFrame:
    rows = [{"subject": s.id, "group": s.group, **summary(s.choice, s.net)} for s in subs]
    return pd.DataFrame(rows).set_index("subject")


def learning_curves(subs: list[Subject], block: int = 10) -> pd.DataFrame:
    """Share of good-deck (C, D) choices per block of trials, by group."""
    rows = []
    for s in subs:
        good = np.isin(s.choice, [2, 3])
        for b in range(0, len(good), block):
            rows.append({"group": s.group, "subject": s.id, "block": b // block + 1,
                         "good": good[b:b + block].mean()})
    return pd.DataFrame(rows)
