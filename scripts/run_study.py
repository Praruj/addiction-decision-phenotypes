"""Run the full human study and write outputs/results.json plus figures.

    python scripts/run_study.py            # full run (~10-20 min on a laptop CPU)
    python scripts/run_study.py --quick    # smoke test
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import behavior, models, rnn  # noqa: E402
from src.data import load_trials, subjects  # noqa: E402
from src.plots import (plot_classification, plot_forecast, plot_learning,  # noqa: E402
                       plot_params)

OUT = ROOT / "outputs"
FIG = OUT / "figures"
GROUPS = ["control", "amphetamine", "heroin"]
SPLIT = 60  # forecast trials 61-100 from trials 1-60
torch.set_num_threads(max(1, torch.get_num_threads()))


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def holm(p: list[float]) -> list[float]:
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj.tolist()


def group_tests(df: pd.DataFrame, cols: list[str], groups: pd.Series) -> list[dict]:
    """Kruskal-Wallis across 3 groups, then each user group vs control (rank-biserial r)."""
    rows = []
    for c in cols:
        x = {g: df.loc[groups == g, c].dropna().to_numpy() for g in GROUPS}
        kw = stats.kruskal(*x.values())
        row = {"feature": c, "kruskal_H": float(kw.statistic), "kruskal_p": float(kw.pvalue)}
        for g in ("amphetamine", "heroin"):
            u = stats.mannwhitneyu(x[g], x["control"], alternative="two-sided")
            row[f"{g}_vs_control_p"] = float(u.pvalue)
            row[f"{g}_vs_control_r"] = float(2 * u.statistic / (len(x[g]) * len(x["control"])) - 1)
            row[f"{g}_median"] = float(np.median(x[g]))
        row["control_median"] = float(np.median(x["control"]))
        rows.append(row)
    adj = holm([r["kruskal_p"] for r in rows])
    for r, a in zip(rows, adj):
        r["kruskal_p_holm"] = a
    return rows


def boot_ci(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(n)]
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    steps = 150 if a.quick else 1200
    restarts = 3 if a.quick else 12
    gru_seeds = (0,) if a.quick else (0, 1, 2)
    FIG.mkdir(parents=True, exist_ok=True)
    res: dict = {}

    df = load_trials()
    subs = subjects(df)
    ids = [s.id for s in subs]
    grp = pd.Series([s.group for s in subs], index=ids)
    y_sub = (grp != "control").astype(int).to_numpy()
    b = models.to_batch(subs)
    res["n"] = {g: int((grp == g).sum()) for g in GROUPS}
    res["n_choices"] = int(len(df))
    log(f"{len(subs)} people, {len(df)} choices")

    # ---------------------------------------------------------------- 1. behaviour
    beh = behavior.table(subs)
    beh.to_csv(OUT / "behaviour.csv")
    curves = behavior.learning_curves(subs)
    plot_learning(curves, FIG / "learning_curves.png")
    bcols = ["net_score", "net_block1", "net_block5", "p_A", "p_B", "p_C", "p_D",
             "win_stay", "lose_shift", "switch_rate", "money"]
    res["behaviour_tests"] = group_tests(beh, bcols, grp)
    res["behaviour_means"] = beh.groupby("group")[bcols].mean().round(3).to_dict("index")

    # ---------------------------------------------------------------- 2. model fits
    fits, bics = {}, {}
    for m in models.SPECS:
        t0 = time.time()
        f = models.fit(m, b, restarts=restarts, steps=steps)
        fits[m] = f
        bics[m] = models.bic(f)
        log(f"fit {m} on all trials: mean BIC {bics[m].mean():.1f} ({time.time() - t0:.0f}s)")
        pd.DataFrame(f.params, index=ids, columns=models.SPECS[m].params).assign(
            group=grp.values).to_csv(OUT / f"params_{m}.csv")
        pd.DataFrame(f.z, index=ids, columns=models.SPECS[m].params).to_csv(OUT / f"params_{m}_z.csv")
    B = pd.DataFrame(bics, index=ids)
    best_counts = B.idxmin(1).groupby(grp).value_counts().unstack(fill_value=0)
    res["model_comparison"] = {
        "mean_bic": B.mean().round(2).to_dict(),
        "mean_bic_by_group": B.groupby(grp).mean().round(2).to_dict("index"),
        "best_model_counts_by_group": best_counts.to_dict("index"),
    }
    best_model = B.mean().idxmin()
    res["best_model"] = best_model
    log(f"best model overall: {best_model}")
    param_tests = {}
    for m, f in fits.items():
        P = pd.DataFrame(f.params, index=ids, columns=models.SPECS[m].params)
        param_tests[m] = group_tests(P, list(P.columns), grp)
    res["param_tests"] = param_tests
    plot_params(pd.DataFrame(fits["orl"].params, index=ids, columns=models.SPECS["orl"].params),
                grp, FIG / "orl_params.png")

    # ---------------------------------------------------------------- 3. forecasting
    # Cognitive models: fit each person's trials 1-60, predict 61-100 one step ahead.
    train_mask = torch.zeros_like(b.mask)
    train_mask[:, :SPLIT] = True
    test = b.mask.clone()
    test[:, :SPLIT] = False
    test_np = test.numpy()
    choice = b.choice.numpy()

    def per_person_nll(p: np.ndarray) -> np.ndarray:
        lp = np.log(np.take_along_axis(p, choice[..., None], -1)[..., 0] + 1e-12)
        return -(lp * test_np).sum(1) / test_np.sum(1)

    def per_person_acc(p: np.ndarray) -> np.ndarray:
        return ((p.argmax(-1) == choice) * test_np).sum(1) / test_np.sum(1)

    probs = {"chance": np.full(choice.shape + (4,), 0.25)}
    freq = np.stack([np.bincount(c[:SPLIT], minlength=4) + 1 for c in choice]).astype(float)
    probs["static_preference"] = np.repeat((freq / freq.sum(1, keepdims=True))[:, None], 100, 1)
    for m in models.SPECS:
        f = models.fit(m, b, fit_mask=train_mask, restarts=restarts, steps=steps)
        probs[m] = models.predict_proba(m, f.params, b)
        log(f"forecast {m}: NLL {per_person_nll(probs[m]).mean():.3f}")

    # GRU: 5-fold over people; it never sees the test person during training.
    p_gru = np.zeros_like(probs["chance"])
    emb = None
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    for k, (tr, te) in enumerate(skf.split(ids, grp.values)):
        net = rnn.train(models.Batch(b.choice[tr], b.out[tr], b.mask[tr]), seeds=gru_seeds,
                        epochs=60 if a.quick else 600)
        bte = models.Batch(b.choice[te], b.out[te], b.mask[te])
        p_gru[te] = net.proba(bte)
        e = net.embed(bte)
        if emb is None:
            emb = np.zeros((len(ids), e.shape[1]))
        emb[te] = e
        log(f"GRU fold {k + 1}/5 done")
    probs["gru"] = p_gru
    log(f"forecast gru: NLL {per_person_nll(p_gru).mean():.3f}")

    fc = {}
    for name, p in probs.items():
        nll = per_person_nll(p)
        acc = per_person_acc(p)
        fc[name] = {"nll": float(nll.mean()), "nll_ci": boot_ci(nll),
                    "accuracy": float(acc.mean()), "accuracy_ci": boot_ci(acc),
                    "nll_by_group": {g: float(nll[grp.values == g].mean()) for g in GROUPS}}
    best_cog = min(models.SPECS, key=lambda m: fc[m]["nll"])
    d = per_person_nll(probs[best_cog]) - per_person_nll(p_gru)
    fc["gru_minus_best_theory"] = {"model": best_cog, "nll_gain": float(d.mean()),
                                   "ci": boot_ci(d), "share_people_gru_better": float((d > 0).mean()),
                                   "wilcoxon_p": float(stats.wilcoxon(d).pvalue)}
    res["forecast"] = fc
    plot_forecast(fc, FIG / "forecast.png")

    # ---------------------------------------------------------------- 4. classification
    feats = {
        "behaviour": beh[bcols].fillna(beh[bcols].median()).to_numpy(),
        "pvl_delta params": fits["pvl_delta"].z,
        "vpp params": fits["vpp"].z,
        "orl params": fits["orl"].z,
        "gru embedding": emb,
    }
    feats["behaviour + orl"] = np.hstack([feats["behaviour"], feats["orl params"]])
    feats["all"] = np.hstack([feats["behaviour"], feats["orl params"], feats["vpp params"], emb])
    tasks = {
        "any substance vs control": (grp.values != "control", np.ones(len(ids), bool)),
        "heroin vs control": (grp.values == "heroin", grp.values != "amphetamine"),
        "amphetamine vs control": (grp.values == "amphetamine", grp.values != "heroin"),
        "heroin vs amphetamine": (grp.values == "heroin", grp.values != "control"),
    }
    repeats = 3 if a.quick else 20
    clf_res = {}
    for task, (yall, keep) in tasks.items():
        y = yall[keep].astype(int)
        clf_res[task] = {}
        for fname, X in feats.items():
            Xk = X[keep]
            aucs = []
            for r in range(repeats):
                oof = np.zeros(len(y))
                for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(Xk, y):
                    clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
                    clf.fit(Xk[tr], y[tr])
                    oof[te] = clf.predict_proba(Xk[te])[:, 1]
                aucs.append(roc_auc_score(y, oof))
            clf_res[task][fname] = {"auc": float(np.mean(aucs)), "auc_sd": float(np.std(aucs))}
        log(f"{task}: " + ", ".join(f"{k} {v['auc']:.2f}" for k, v in clf_res[task].items()))

    # permutation test for the headline task with its best feature set
    task = "any substance vs control"
    best_f = max(clf_res[task], key=lambda k: clf_res[task][k]["auc"])
    X, y = feats[best_f], y_sub
    rng = np.random.default_rng(0)
    obs = clf_res[task][best_f]["auc"]
    null = []
    for i in range(30 if a.quick else 500):
        yp = rng.permutation(y)
        oof = np.zeros(len(y))
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=i).split(X, yp):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=2000))
            clf.fit(X[tr], yp[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        null.append(roc_auc_score(yp, oof))
    res["classification"] = clf_res
    res["classification_permutation"] = {"task": task, "features": best_f, "auc": obs,
                                         "p": float((np.sum(np.array(null) >= obs) + 1) / (len(null) + 1)),
                                         "null_mean": float(np.mean(null))}
    plot_classification(clf_res, FIG / "classification.png")

    # keep what the LLM analysis needs: humans' behaviour + ORL/VPP params
    np.save(OUT / "gru_embedding.npy", emb)
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    log("wrote outputs/results.json")


if __name__ == "__main__":
    main()
