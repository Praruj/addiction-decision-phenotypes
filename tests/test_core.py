import numpy as np
import torch

from src import behavior, models, rnn
from src.data import Subject, load_trials, payoff_schedule, schedule_consistency, subjects


def test_data_loads_three_groups():
    df = load_trials()
    assert set(df.group) == {"control", "amphetamine", "heroin"}
    assert df.subject.nunique() == 129
    assert (df.loss <= 0).all() and df.deck.between(0, 3).all()


def test_payoffs_are_a_fixed_schedule():
    assert schedule_consistency() == 1.0
    s = payoff_schedule()
    assert len(s) == 4 and all(len(v) >= 40 for v in s.values())


def test_behaviour_summary():
    c = np.array([0, 0, 2, 3] * 25)
    f = behavior.summary(c, np.where(c < 2, -100.0, 50.0))
    assert f["net_score"] == 0 and abs(f["p_A"] - 0.5) < 1e-9


def _batch(n=6, T=100, seed=0):
    rng = np.random.default_rng(seed)
    subs = [Subject(str(i), "control", rng.integers(0, 4, T), rng.uniform(40, 120, T),
                    -rng.choice([0, 0, 0, 250], T).astype(float)) for i in range(n)]
    return models.to_batch(subs)


def test_model_logits_are_causal():
    b = _batch()
    for m, spec in models.SPECS.items():
        th = models.constrain(spec, torch.zeros(6, spec.n, dtype=torch.float64))
        l1 = models.logits(m, th, b)
        b2 = models.Batch(b.choice.clone(), b.out.clone(), b.mask)
        b2.choice[:, 50:] = (b2.choice[:, 50:] + 1) % 4
        b2.out[:, 50:] = 0.3
        l2 = models.logits(m, th, b2)
        assert torch.allclose(l1[:, :51], l2[:, :51]), m


def test_orl_parameter_recovery():
    """Simulate players with known ORL parameters on the real task, then refit."""
    sched = payoff_schedule()
    rng = np.random.default_rng(1)
    true = np.column_stack([rng.uniform(0.05, 0.4, 16), rng.uniform(0.05, 0.4, 16),
                            rng.uniform(0.2, 2, 16), rng.uniform(-3, 3, 16), rng.uniform(-3, 3, 16)])
    subs = []
    for i, th in enumerate(true):
        h = models.simulate("orl", th, sched, T=100, seed=i)
        subs.append(Subject(str(i), "sim", np.array([x[0] for x in h]),
                            np.array([x[1] for x in h]), np.array([x[2] for x in h])))
    f = models.fit("orl", models.to_batch(subs), restarts=4, steps=400)
    r_betaF = np.corrcoef(true[:, 3], f.params[:, 3])[0, 1]
    r_betaP = np.corrcoef(true[:, 4], f.params[:, 4])[0, 1]
    assert r_betaF > 0.5 and r_betaP > 0.5


def test_gru_inputs_are_shifted():
    b = _batch(2)
    x = rnn.inputs(b)
    assert x.shape == (2, 100, rnn.IN_DIM)
    assert x[:, 0, :8].abs().sum() == 0 and (x[:, 0, -1] == 1).all()
    onehot_prev = x[:, 1:, :4].argmax(-1)
    assert torch.equal(onehot_prev, b.choice[:, :-1])


def test_gru_trains_quickly():
    b = models.to_batch(subjects()[:20])
    net = rnn.train(b, seeds=(0,), epochs=15)
    p = net.proba(b)
    assert p.shape == (20, 100, 4) and np.allclose(p.sum(-1), 1, atol=1e-5)
