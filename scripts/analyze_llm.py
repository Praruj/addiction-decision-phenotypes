"""Place LLM players in the human decision space.

Reads outputs/llm/*.jsonl (from `python -m src.llm_agent` on a PC or the Colab notebook),
fits the same cognitive models used for people, and asks a classifier trained only on
people: does this player look more like a control or like a substance user?

    python scripts/analyze_llm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import behavior, models  # noqa: E402
from src.data import Subject, subjects  # noqa: E402
from src.plots import plot_learning, plot_params  # noqa: E402

OUT = ROOT / "outputs"
LABEL = {"direct": "LLM direct", "cot": "LLM chain-of-thought"}
BCOLS = ["net_score", "net_block1", "net_block5", "p_A", "p_B", "p_C", "p_D",
         "win_stay", "lose_shift", "switch_rate", "money"]


def load_games() -> list[dict]:
    games = []
    for f in sorted((OUT / "llm").glob("*.jsonl")):
        games += [json.loads(line) for line in f.read_text(encoding="utf-8").splitlines() if line]
    return games


def main() -> None:
    games = load_games()
    if not games:
        sys.exit("no games in outputs/llm/ - run `python -m src.llm_agent` first")
    agents = []
    for i, g in enumerate(games):
        tr = g["trials"]
        short = g["model"].split("/")[-1]
        agents.append(Subject(f"{short}-{g['condition']}-{i}", LABEL[g["condition"]],
                              np.array([t["deck"] for t in tr]),
                              np.array([t["gain"] for t in tr], float),
                              np.array([t["loss"] for t in tr], float)))
    humans = subjects()
    beh_h = behavior.table(humans)
    beh_a = behavior.table(agents)
    beh_a["model"] = [a.id.rsplit("-", 2)[0] for a in agents]

    # same model fits as for people
    ba = models.to_batch(agents)
    fit_a = models.fit("orl", ba)
    params_h = pd.read_csv(OUT / "params_orl.csv", index_col=0)
    zs_h = pd.read_csv(OUT / "params_orl_z.csv", index_col=0).loc[beh_h.index].to_numpy()
    Pa = pd.DataFrame(fit_a.params, index=[a.id for a in agents], columns=models.SPECS["orl"].params)

    # human-trained classifier: substance user vs control, behaviour + ORL parameters
    Xh = np.hstack([beh_h[BCOLS].fillna(beh_h[BCOLS].median()).to_numpy(), zs_h])
    yh = (beh_h["group"] != "control").astype(int).to_numpy()
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000)).fit(Xh, yh)
    Xa = np.hstack([beh_a[BCOLS].fillna(beh_h[BCOLS].median()).to_numpy(), fit_a.z])
    beh_a["p_substance_like"] = clf.predict_proba(Xa)[:, 1]
    for c in Pa.columns:
        beh_a[c] = Pa[c].to_numpy()
    beh_a.to_csv(OUT / "llm_agents.csv")

    summary = {}
    for (m, g), d in beh_a.groupby(["model", "group"]):
        summary[f"{m} | {g}"] = {
            "games": int(len(d)), "net_score": float(d.net_score.mean()),
            "money": float(d.money.mean()), "p_substance_like": float(d.p_substance_like.mean()),
            "win_stay": float(d.win_stay.mean()), "lose_shift": float(d.lose_shift.mean()),
            **{c: float(d[c].median()) for c in Pa.columns}}
    hum = beh_h.groupby("group")[["net_score", "money", "win_stay", "lose_shift"]].mean()
    summary["humans"] = hum.round(3).to_dict("index")
    p_h = clf.predict_proba(Xh)[:, 1]
    summary["humans_p_substance_like_in_sample"] = {
        g: float(p_h[beh_h.group.to_numpy() == g].mean()) for g in ["control", "amphetamine", "heroin"]}
    (OUT / "llm_results.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    curves = pd.concat([behavior.learning_curves(humans), behavior.learning_curves(agents)])
    plot_learning(curves[curves.group.isin(["control", "heroin"])], OUT / "figures" / "llm_learning.png",
                  extra=curves[curves.group.str.startswith("LLM")])
    allP = pd.concat([params_h[list(Pa.columns)], Pa])
    grp = pd.Series(list(params_h["group"]) + [a.group for a in agents], index=allP.index)
    order = ["control", "amphetamine", "heroin"] + [g for g in LABEL.values() if g in set(grp)]
    plot_params(allP, grp, OUT / "figures" / "llm_orl_params.png", order=order,
                title="ORL parameters: people vs LLM players")


if __name__ == "__main__":
    main()
