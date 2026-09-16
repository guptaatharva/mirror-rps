"""
analysis.py
-----------
Shared, side-effect-free analysis + simulation helpers used by BOTH the
offline report script (evaluate.py) and the live GUI (app.py), so the two
never drift out of sync and so importing this module never reseeds the
global random state (that seeding lives only in evaluate.py's __main__,
for reproducible reports -- the live app must stay genuinely random).

  - behavioral_features(history) : turn a played session into the fixed-length
    feature vector that PCA is applied to.
  - fit_pca(feature_rows)        : standardize + PCA -> 2D.
  - RandomAgent / SinglePredictorAgent : baseline "agents" for comparison.
  - run_game(...)                : play one simulated archetype vs one agent kind.
  - exact_binomial_pvalue(...)   : significance of a win rate vs the 33% baseline.
"""

import math
import random

import pandas as pd
from sklearn.decomposition import PCA

from predictors import build_predictor_bank, counter_move, outcome, MOVES
from agent import MirrorAgent, UCB1BanditAgent

FEATURE_COLUMNS = [
    "freq_rock", "freq_paper", "freq_scissors",
    "stay_after_win_rate", "shift_after_loss_rate", "repeat_rate",
]


def behavioral_features(history):
    """Turn a played session into a fixed-length feature vector describing
    HOW the player behaved -- this is the feature space PCA is applied to."""
    moves = [h["human"] for h in history]
    n = len(moves)
    if n == 0:
        return {
            "freq_rock": 1 / 3, "freq_paper": 1 / 3, "freq_scissors": 1 / 3,
            "stay_after_win_rate": 0.5, "shift_after_loss_rate": 0.5, "repeat_rate": 0.5,
        }
    counts = {m: moves.count(m) / n for m in MOVES}

    stay_after_win, win_rounds = 0, 0
    shift_after_loss, loss_rounds = 0, 0
    repeats = 0
    for i in range(1, n):
        prev, curr = history[i - 1], history[i]
        if prev["human"] == curr["human"]:
            repeats += 1
        if prev["result"] == "human":
            win_rounds += 1
            if curr["human"] == prev["human"]:
                stay_after_win += 1
        elif prev["result"] == "agent":
            loss_rounds += 1
            if curr["human"] != prev["human"]:
                shift_after_loss += 1

    return {
        "freq_rock": counts["rock"],
        "freq_paper": counts["paper"],
        "freq_scissors": counts["scissors"],
        "stay_after_win_rate": stay_after_win / win_rounds if win_rounds else 0.5,
        "shift_after_loss_rate": shift_after_loss / loss_rounds if loss_rounds else 0.5,
        "repeat_rate": repeats / (n - 1) if n > 1 else 0.5,
    }


def fit_pca(feature_rows):
    """feature_rows: list of dicts (as produced by behavioral_features).
    Returns (coords [n x 2], explained_variance_ratio [2])."""
    feat_df = pd.DataFrame(feature_rows)[FEATURE_COLUMNS]
    X = feat_df.values
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_


# ---------------------------------------------------------------------
# Baseline agents to compare against MirrorAgent (rubric: "compare
# different algorithms / modelling approaches")
# ---------------------------------------------------------------------

class RandomAgent:
    """No learning at all -- the floor every real approach must beat."""
    def choose_move(self, history):
        return random.choice(MOVES)


class SinglePredictorAgent:
    """Always trusts ONE fixed predictor, no meta-learning. Used to show that
    the Q-learning ensemble beats any single fixed heuristic on its own."""
    def __init__(self, predictor):
        self.predictor = predictor

    def choose_move(self, history):
        predicted = self.predictor.predict(history)
        return counter_move(predicted)

    def update(self, history):
        self.predictor.update(history)


def run_game(agent_kind, archetype_fn, n_rounds, agent_config=None):
    """Plays n_rounds of one archetype against one agent kind.
    agent_kind: "mirror" | "ucb1" | "random" | a predictor name (single-predictor baseline).
    Returns (win_rate_curve, history)."""
    history = []
    wins = 0
    win_rate_curve = []

    if agent_kind == "mirror":
        active_agent = MirrorAgent(seed=None, **(agent_config or {}))
    elif agent_kind == "ucb1":
        active_agent = UCB1BanditAgent(seed=None, **(agent_config or {}))
    elif agent_kind == "random":
        active_agent = RandomAgent()
    else:  # single-predictor baselines: agent_kind is the predictor's display name
        bank = {p.name: p for p in build_predictor_bank()}
        active_agent = SinglePredictorAgent(bank[agent_kind])

    for r in range(n_rounds):
        human_move = archetype_fn(history)

        if agent_kind in ("mirror", "ucb1"):
            agent_move, _, _ = active_agent.choose_move()
        else:
            agent_move = active_agent.choose_move(history)

        result = outcome(human_move, agent_move)  # 'human' | 'agent' | 'tie'
        history.append({"human": human_move, "agent": agent_move, "result": result})

        if agent_kind in ("mirror", "ucb1"):
            active_agent.learn(human_move, agent_move)
        elif agent_kind != "random":
            active_agent.update(history)

        if result == "agent":
            wins += 1
        win_rate_curve.append(wins / (r + 1))

    return win_rate_curve, history


def exact_binomial_pvalue(wins, n, p0=1 / 3, alternative="greater"):
    """Exact binomial p-value without requiring scipy."""
    def probability(k):
        return math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k))

    if alternative == "greater":
        return sum(probability(k) for k in range(wins, n + 1))
    observed_probability = probability(wins)
    return min(1.0, sum(
        probability(k) for k in range(n + 1)
        if probability(k) <= observed_probability + 1e-15
    ))
