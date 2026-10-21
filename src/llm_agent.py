"""Let an open LLM play the same Iowa Gambling Task, with and without chain-of-thought.

The agent gets the standard IGT instructions, its current balance and only its last
`memory` draws (people cannot recall all 100 either). Two conditions:

* direct - the model answers at once.
* cot    - the model first reasons in a few sentences (chain-of-thought), then answers.

In both conditions the final answer is read the same way: after the prompt (and any
reasoning) we append "I choose deck" and take the model's probabilities for A/B/C/D,
then sample. So nothing has to be parsed from free text, and the two conditions differ
only in whether reasoning comes first. Games are saved as JSON lines so the same
cognitive models used for people can be fitted to them.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .data import DECKS, payoff_schedule
from .models import draw

INSTRUCTIONS = (
    "You are taking part in a card game experiment. There are four decks of cards: "
    "A, B, C and D. Each time you pick a card you win some money, and sometimes you "
    "also lose some money. Some decks are worse than others. You do not know in advance "
    "which ones. Your goal is to win as much money as possible. You start with $2000 "
    "and there are 100 picks in total."
)


def prompt_text(t: int, balance: float, hist: list[dict], memory: int, cot: bool) -> str:
    lines = [INSTRUCTIONS, "", f"Pick {t + 1} of 100. Your balance is ${balance:.0f}."]
    recent = hist[-memory:]
    if recent:
        lines.append(f"Your last {len(recent)} picks:")
        for h in recent:
            res = f"won ${h['gain']:.0f}"
            if h["loss"] < 0:
                res += f" and lost ${-h['loss']:.0f}"
            lines.append(f"- pick {h['t'] + 1}: deck {DECKS[h['deck']]}, {res}")
    else:
        lines.append("This is your first pick.")
    if cot:
        lines.append("\nThink step by step in two or three short sentences about which "
                     "deck is best right now. Then say which deck you choose.")
    else:
        lines.append("\nWhich deck do you choose? Answer with one letter.")
    return "\n".join(lines)


class Agent:
    def __init__(self, model_id: str, device: str | None = None, dtype: str = "auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_id)
        kw = {}
        if dtype != "auto":
            kw["torch_dtype"] = getattr(torch, dtype)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()
        ids = [self.tok.encode(" " + d, add_special_tokens=False)[-1] for d in DECKS]
        self.letter_ids = ids

    def _chat(self, user: str) -> str:
        msgs = [{"role": "user", "content": user}]
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def choose(self, user: str, cot: bool, rng: np.random.Generator,
               max_reason_tokens: int = 80, temperature: float = 1.0) -> tuple[int, np.ndarray, str]:
        torch = self.torch
        text = self._chat(user)
        reasoning = ""
        if cot:
            enc = self.tok(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                g = self.model.generate(**enc, max_new_tokens=max_reason_tokens, do_sample=True,
                                        temperature=0.7, top_p=0.95,
                                        pad_token_id=self.tok.eos_token_id)
            reasoning = self.tok.decode(g[0, enc.input_ids.shape[1]:], skip_special_tokens=True)
            # keep the reasoning, drop any answer it already wrote, then ask for the answer
            for stop in ("I choose deck", "I choose", "Choice:", "choose deck"):
                if stop in reasoning:
                    reasoning = reasoning[:reasoning.index(stop)]
            reasoning = reasoning.strip()
            text = text + reasoning + "\n"
        enc = self.tok(text + "I choose deck", return_tensors="pt").to(self.device)
        with torch.no_grad():
            logit = self.model(**enc).logits[0, -1, self.letter_ids].float()
        p = torch.softmax(logit / temperature, -1).cpu().numpy().astype(float)
        p = p / p.sum()
        return int(rng.choice(4, p=p)), p, reasoning


def play(agent: Agent, cot: bool, seed: int, memory: int = 10, T: int = 100,
         schedule: dict | None = None) -> dict:
    schedule = schedule or payoff_schedule()
    rng = np.random.default_rng(seed)
    balance, draws, hist = 2000.0, [0, 0, 0, 0], []
    t0 = time.time()
    for t in range(T):
        d, p, why = agent.choose(prompt_text(t, balance, hist, memory, cot), cot, rng)
        gain, loss = draw(schedule, d, draws[d])
        draws[d] += 1
        balance += gain + loss
        hist.append({"t": t, "deck": d, "gain": gain, "loss": loss, "p": p.round(4).tolist(),
                     "reasoning": why})
    return {"condition": "cot" if cot else "direct", "seed": seed, "memory": memory,
            "seconds": round(time.time() - t0, 1), "final_balance": balance, "trials": hist}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--games", type=int, default=4, help="games per condition")
    ap.add_argument("--conditions", default="direct,cot")
    ap.add_argument("--memory", type=int, default=10)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", default="outputs/llm/games.jsonl")
    ap.add_argument("--dtype", default="auto")
    a = ap.parse_args()
    agent = Agent(a.model, dtype=a.dtype)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sched = payoff_schedule()
    for cond in a.conditions.split(","):
        for g in range(a.games):
            game = play(agent, cond == "cot", seed=1000 + g, memory=a.memory, T=a.trials,
                        schedule=sched)
            game["model"] = a.model
            with out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(game) + "\n")
            print(f"{a.model} {cond} game {g}: balance {game['final_balance']:.0f} "
                  f"in {game['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
