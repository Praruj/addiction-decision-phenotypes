"""Load the Ahn et al. (2014) Iowa Gambling Task data.

Three groups played the same modified IGT (Bechara et al., 2001) for 100 trials:
healthy controls, "pure" amphetamine users and "pure" heroin users, all in protracted
abstinence. Each row is one card draw.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
GROUPS = {"control": "IGTdata_healthy_control.txt",
          "amphetamine": "IGTdata_amphetamine.txt",
          "heroin": "IGTdata_heroin.txt"}
DECKS = "ABCD"
N_TRIALS = 100


def load_trials(raw_dir: Path = RAW) -> pd.DataFrame:
    """All trials as one tidy frame: subject, group, trial, deck (0-3), gain, loss, net."""
    frames = []
    for group, fname in GROUPS.items():
        df = pd.read_csv(raw_dir / fname, sep="\t")
        df["group"] = group
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"subjID": "subject"})
    df["subject"] = df["group"].str[0] + df["subject"].astype(str)
    df["deck"] = df["deck"].astype(int) - 1
    df["loss"] = -df["loss"].abs()
    df["net"] = df["gain"] + df["loss"]
    return df[["subject", "group", "trial", "deck", "gain", "loss", "net"]]


@dataclass
class Subject:
    id: str
    group: str
    choice: np.ndarray  # (T,) deck index 0-3
    gain: np.ndarray    # (T,)
    loss: np.ndarray    # (T,) <= 0

    @property
    def net(self) -> np.ndarray:
        return self.gain + self.loss


def subjects(df: pd.DataFrame | None = None) -> list[Subject]:
    df = load_trials() if df is None else df
    out = []
    for sid, g in df.sort_values(["subject", "trial"]).groupby("subject", sort=False):
        out.append(Subject(sid, g["group"].iat[0], g["deck"].to_numpy(),
                           g["gain"].to_numpy(float), g["loss"].to_numpy(float)))
    return out


def payoff_schedule(df: pd.DataFrame | None = None) -> dict[int, list[tuple[float, float]]]:
    """Reconstruct the fixed card sequence of each deck.

    In the IGT the k-th card drawn from a deck always pays the same, whoever draws it.
    For every deck and draw index we take the most common (gain, loss) across subjects
    and report how consistent that is (see `schedule_consistency`).
    """
    df = load_trials() if df is None else df
    df = df.sort_values(["subject", "trial"]).copy()
    df["k"] = df.groupby(["subject", "deck"]).cumcount()
    sched = {}
    for d, g in df.groupby("deck"):
        rows = []
        for k, gk in g.groupby("k"):
            rows.append(tuple(gk[["gain", "loss"]].value_counts().index[0]))
        sched[int(d)] = rows
    return sched


def schedule_consistency(df: pd.DataFrame | None = None) -> float:
    """Share of draws whose payoff equals the modal payoff for that (deck, draw index)."""
    df = load_trials() if df is None else df
    df = df.sort_values(["subject", "trial"]).copy()
    df["k"] = df.groupby(["subject", "deck"]).cumcount()
    mode = (df.groupby(["deck", "k", "gain", "loss"]).size().rename("n").reset_index()
              .sort_values("n").groupby(["deck", "k"]).tail(1))
    m = df.merge(mode, on=["deck", "k"], suffixes=("", "_m"))
    return float(((m.gain == m.gain_m) & (m.loss == m.loss_m)).mean())
