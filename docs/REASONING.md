# Research reasoning, step by step

This file records the chain of thought behind the project: each design decision, the
alternatives considered, and why one was picked. It was written *before* the results were in,
and the decisions below were not changed after seeing them.

## 1. What question is worth asking?

- Addiction is often described as a disorder of **decision-making**: choosing a reward now
  over a larger loss later. The classic lab measure of this is the **Iowa Gambling Task (IGT)**.
  Two decks (A, B) pay a lot but lose more. Two (C, D) pay less but win in the long run.
- Saying "users pick bad decks more" is old news. The more useful question is *why*:
  do they learn more slowly from losses, care less about losses, or just repeat what they did
  last time? Two people can get the same score for different reasons.
- **Computational psychiatry** answers this by fitting a learning model to every choice. Its
  parameters (learning rate, loss aversion, perseverance) are a "decision phenotype".
- That leads to three questions this project can test on real data:
  1. **Theory vs. AI.** Hand-built cognitive models encode psychology theory. A neural network
     learns from data alone. Which predicts what a *new* person will do next?
  2. **Phenotype vs. score.** Can the model parameters tell substance users apart from
     controls better than plain behaviour summaries can? And does heroin differ from
     amphetamine?
  3. **Where do AI agents sit?** If an LLM plays the same task, does it decide like a
     control or like a substance user, and does **chain-of-thought** reasoning change that?

## 2. Which data?

- Needed: trial-by-trial choices (not only total scores) from clinically defined groups,
  under an open licence.
- Picked **Ahn et al. (2014)**: 48 controls, 38 "pure" amphetamine and 43 "pure" heroin
  users in protracted abstinence (Bulgaria). That is 129 people × 100 trials, about 12,900
  choices, CC BY 4.0 on figshare.
- Why it fits: "pure" users have a single drug, which is rare and lets the two drugs be
  compared without polysubstance confounds. Being abstinent means the results describe traits,
  not intoxication.
- Limitation, accepted up front: 129 people is small. So: simple classifiers, repeated
  cross-validation, permutation tests, and confidence intervals, and no claims of
  diagnosis.

## 3. Which cognitive models?

- The three standard IGT models from hBayesDM, so results compare with the literature:
  **PVL-Delta** (4 parameters), **VPP** (8) and **ORL** (5).
- ORL is the main one for interpretation. It splits learning from rewards (`Arew`) from
  learning from punishments (`Apun`), and has a win-frequency weight (`betaF`) and a
  perseverance weight (`betaP`). Each maps to an addiction hypothesis.
- Fitting: per-person penalised maximum likelihood, written in PyTorch so all 129 people ×
  12 random restarts fit in one batch with exact gradients. Hierarchical Bayes would be better
  for small N but needs Stan. Penalised MLE with a weak prior is a reasonable middle ground.
  A parameter-recovery test checks that the fitter finds known parameters in simulated data.

## 4. How to compare theory and AI fairly?

- The trap: cognitive models are fitted to a person's own choices, so scoring them on the
  same choices flatters them.
- The rule: **forecast the future**. Every model sees trials 1–60 of a person and predicts
  each of trials 61–100 one step ahead (it sees the real outcome before the next prediction).
  - Cognitive models: fitted to that person's trials 1–60 only.
  - GRU: trained on *other* people (5-fold over people) and only reads this person's history.
    It has to adapt in context, with no per-person training.
- Baselines: chance (log 4 = 1.386 nats per choice) and "static preference" (this person's
  deck frequencies in trials 1–60). Any model that can't beat static preference has not
  learned the learning.
- Metric: negative log-likelihood per choice (and top-1 accuracy). Bootstrap CI over
  people, and a Wilcoxon test on the per-person difference.

## 5. How to test "can we detect addiction from decisions"?

- Features compared on equal terms: behaviour summaries, each model's parameters, the GRU's
  hidden-state embedding (out-of-fold, so the network never trained on that person), and
  combinations.
- Classifier: L2 logistic regression only. With ~90 people per task a flexible model would
  overfit, and the point is to compare feature sets, not classifiers.
- 20 × repeated stratified 5-fold CV for AUC. A 500-permutation test for the headline
  result. Four tasks: any substance, heroin, amphetamine vs. control, and heroin vs. amphetamine.

## 6. The LLM experiment

- The IGT pays exactly the same k-th card from a deck to everyone. The data confirms this
  (100% consistent), so the real task can be rebuilt and given to an AI agent.
- Fairness choices: the standard instructions, the balance, and only the **last 10 draws**
  (people don't remember all 100). No per-deck statistics are given, since that would turn it
  into an easy bandit.
- Two conditions that differ in **one thing only**: answer directly, or reason in 2–3
  sentences first (chain-of-thought). In both, the answer is read from the model's
  probabilities over A/B/C/D after "I choose deck", so there is no fragile text parsing.
- Then the agent's 100 choices go through the *same* model fits and the *same* human-trained
  classifier: "how substance-user-like is this player?".
- On the laptop CPU only a small model (Qwen2.5-0.5B) is practical, as a pilot. The Colab
  notebook runs a 3B model with 10 games per condition on a free GPU.

## 7. Ethics

- The data is de-identified and openly licensed. Group labels are clinical research
  categories. Nothing here is a diagnostic tool, and the README says so.
- Results are reported in aggregate. Small-sample uncertainty is reported with each number.
