"""Figures. Group colours are slots 1-3 of a colour-blind-validated categorical palette."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

COLORS = {"control": "#2a78d6", "amphetamine": "#eb6834", "heroin": "#1baf7a",
          "LLM direct": "#6250d6", "LLM chain-of-thought": "#e34948"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
GROUPS = ["control", "amphetamine", "heroin"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2,
})


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_learning(curves: pd.DataFrame, path: Path, extra: pd.DataFrame | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    data = curves if extra is None else pd.concat([curves, extra])
    order = [g for g in COLORS if g in set(data.group)]
    for g in order:
        d = data[data.group == g]
        m = d.groupby("block")["good"].agg(["mean", "sem"])
        ls = "--" if g.startswith("LLM") else "-"
        ax.plot(m.index * 10, m["mean"], ls, color=COLORS[g], marker="o", ms=4,
                label=f"{g} (n={d.subject.nunique()})")
        ax.fill_between(m.index * 10, m["mean"] - m["sem"], m["mean"] + m["sem"],
                        color=COLORS[g], alpha=0.12, lw=0)
    ax.axhline(0.5, color=MUTED, lw=1, ls=":")
    ax.set_xlabel("trial")
    ax.set_ylabel("share of good-deck (C, D) picks")
    ax.set_title("Learning to avoid the bad decks")
    ax.set_ylim(0.2 if extra is None else 0, 1)
    ax.legend(loc="lower right", fontsize=8)
    _save(fig, path)


def _strip(ax, P: pd.DataFrame, col: str, groups: pd.Series, order: list[str]) -> None:
    rng = np.random.default_rng(0)
    for i, g in enumerate(order):
        v = P.loc[groups == g, col].to_numpy()
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=14, color=COLORS[g], alpha=0.55,
                   edgecolor=SURFACE, linewidth=0.5)
        ax.hlines(np.median(v), i - 0.3, i + 0.3, color=INK, lw=2)
    short = {"control": "ctrl", "amphetamine": "amph", "heroin": "heroin",
             "LLM direct": "LLM\ndirect", "LLM chain-of-thought": "LLM\nCoT"}
    ax.set_xticks(range(len(order)), [short.get(o, o) for o in order], fontsize=8)
    ax.grid(axis="x", visible=False)


def plot_params(P: pd.DataFrame, groups: pd.Series, path: Path, order=GROUPS,
                title: str = "ORL parameters by group (line = median)") -> None:
    cols = list(P.columns)
    fig, axes = plt.subplots(1, len(cols), figsize=(2.3 * len(cols), 3.4))
    for ax, c in zip(axes, cols):
        _strip(ax, P, c, groups, order)
        ax.set_title(c, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontweight="bold", fontsize=11)
    _save(fig, path)


def plot_forecast(fc: dict, path: Path) -> None:
    names = ["chance", "static_preference", "pvl_delta", "vpp", "orl", "gru"]
    labels = ["chance", "static\npreference", "PVL-Delta", "VPP", "ORL", "GRU\n(AI)"]
    v = [fc[n]["nll"] for n in names]
    lo = [fc[n]["nll"] - fc[n]["nll_ci"][0] for n in names]
    hi = [fc[n]["nll_ci"][1] - fc[n]["nll"] for n in names]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    col = [GRID, GRID, "#9fc3ee", "#9fc3ee", "#9fc3ee", COLORS["control"]]
    ax.bar(range(len(v)), v, color=col, width=0.6, yerr=[lo, hi], ecolor=MUTED, capsize=3)
    for i, x in enumerate(v):
        ax.text(i, x + hi[i] + 0.01, f"{x:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(v)), labels, fontsize=8)
    ax.set_ylabel("NLL per choice, trials 61-100 (lower = better)")
    ax.set_title("Forecasting each person's next 40 choices", pad=14)
    ax.set_ylim(0, max(x + h for x, h in zip(v, hi)) * 1.12)
    ax.grid(axis="x", visible=False)
    _save(fig, path)


def plot_classification(clf: dict, path: Path) -> None:
    tasks = list(clf)
    feats = list(clf[tasks[0]])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    M = np.array([[clf[t][f]["auc"] for f in feats] for t in tasks])
    im = ax.imshow(M, cmap="Blues", vmin=0.4, vmax=0.85, aspect="auto")
    for i in range(len(tasks)):
        for j in range(len(feats)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.7 else INK)
    ax.set_xticks(range(len(feats)), feats, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(tasks)), tasks, fontsize=8)
    ax.grid(False)
    ax.set_title("Cross-validated AUC (0.5 = chance)")
    fig.colorbar(im, ax=ax, fraction=0.03)
    _save(fig, path)
