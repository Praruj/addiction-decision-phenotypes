"""Export the numbers the portfolio case study draws (data.json).

    python scripts/export_site.py ../portfolio/addiction/data.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src import behavior  # noqa: E402
from src.data import subjects  # noqa: E402


def main(out: str) -> None:
    r = json.loads((ROOT / "outputs" / "results.json").read_text())
    curves = behavior.learning_curves(subjects())
    lc = {}
    for g, d in curves.groupby("group"):
        m = d.groupby("block")["good"].agg(["mean", "sem"])
        lc[g] = [{"trial": int(b * 10), "mean": round(float(row["mean"]), 4),
                  "sem": round(float(row["sem"]), 4)} for b, row in m.iterrows()]
    fc = r["forecast"]
    models = ["chance", "static_preference", "pvl_delta", "vpp", "orl", "gru"]
    beh = {t["feature"]: t for t in r["behaviour_tests"]}
    data = {
        "n": r["n"],
        "n_choices": r["n_choices"],
        "forecast": [{"key": m, "nll": round(fc[m]["nll"], 4),
                      "lo": round(fc[m]["nll_ci"][0], 4), "hi": round(fc[m]["nll_ci"][1], 4),
                      "acc": round(fc[m]["accuracy"], 4)} for m in models],
        "gru_vs_theory": fc["gru_minus_best_theory"],
        "learning": lc,
        "classification": {t: {f: round(v["auc"], 3) for f, v in fs.items()}
                           for t, fs in r["classification"].items()},
        "permutation": r["classification_permutation"],
        "medians": {f: {g: beh[f][f"{g}_median"] for g in ("control", "amphetamine", "heroin")}
                    for f in ("net_score", "money", "p_B")},
        "best_model_counts": r["model_comparison"]["best_model_counts_by_group"],
    }
    llm = ROOT / "outputs" / "llm_results.json"
    if llm.exists():
        data["llm"] = json.loads(llm.read_text())
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "outputs" / "site_data.json"))
