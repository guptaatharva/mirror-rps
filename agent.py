"""
agent.py
--------
MirrorAgent: a Q-learning meta-agent sitting on top of the predictor ensemble
in predictors.py.

State  : the recent (last WINDOW_SIZE rounds) accuracy of each predictor,
         discretized into buckets (na / low / med / high) -> a small discrete
         state space the Q-table can actually learn over a short live game.
Action : "which predictor do I trust this round?" (one action per predictor)
Reward : +1 for a round the agent wins, -1 for a round it loses, -0.2 for a tie
         (ties are mildly discouraged so the agent keeps hunting for a real edge)

This is genuine reinforcement learning: the agent is not told which predictor
is best. It discovers that live, purely from round-by-round reward, and the
policy that emerges is specific to whoever it is currently playing against.
"""

import random
import json
import os
from collections import defaultdict

from predictors import build_predictor_bank, counter_move, outcome, MOVES

WINDOW_SIZE = 10


def _bucket(acc):
    if acc < 0.40:
        return "low"
    elif acc < 0.60:
        return "med"
    return "high"


class MirrorAgent:
    def __init__(self, alpha=0.3, gamma=0.7, epsilon=0.15, seed=None,
                 epsilon_start=None, epsilon_min=0.05, epsilon_decay=1.0):
        if seed is not None:
            random.seed(seed)
        self.predictors = build_predictor_bank()
        self.n = len(self.predictors)

        self.accuracy_window = [[] for _ in range(self.n)]
        self.Q = defaultdict(lambda: [0.0] * self.n)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_start = epsilon_start if epsilon_start is not None else epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.history = []          # list of {"human","agent","result"}
        self.round_log = []        # per-round log for charts / narration
        self.trust_log = []        # per-round snapshot of every predictor's windowed accuracy
        self.wins = self.losses = self.ties = 0

        self._last_state = None
        self._last_action = None
        self._last_predictions = None

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ---------- core Q-learning loop ----------

    def _state(self):
        return tuple(
            _bucket(sum(w) / len(w)) if w else "na" for w in self.accuracy_window
        )

    def _choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n)
        qvals = self.Q[state]
        best_q = max(qvals)
        best = [i for i, q in enumerate(qvals) if q == best_q]
        return random.choice(best)

    def choose_move(self):
        """Called BEFORE the human reveals their move. Returns the agent's move."""
        predictions = [p.predict(self.history) for p in self.predictors]
        state = self._state()
        action = self._choose_action(state)
        predicted_human_move = predictions[action]
        agent_move = counter_move(predicted_human_move)

        self._last_state = state
        self._last_action = action
        self._last_predictions = predictions
        return agent_move, predictions, action

    def learn(self, human_move, agent_move):
        """Called AFTER the human reveals their move. Updates predictors + Q-table."""
        result = outcome(human_move, agent_move)  # 'human' | 'agent' | 'tie'

        for i, pred in enumerate(self._last_predictions):
            correct = 1 if pred == human_move else 0
            self.accuracy_window[i].append(correct)
            if len(self.accuracy_window[i]) > WINDOW_SIZE:
                self.accuracy_window[i].pop(0)

        reward = {"agent": 1.0, "tie": -0.2, "human": -1.0}[result]

        self.history.append({"human": human_move, "agent": agent_move, "result": result})
        for p in self.predictors:
            p.update(self.history)

        next_state = self._state()
        old_q = self.Q[self._last_state][self._last_action]
        next_max = max(self.Q[next_state])
        self.Q[self._last_state][self._last_action] = old_q + self.alpha * (
            reward + self.gamma * next_max - old_q
        )

        if result == "agent":
            self.wins += 1
        elif result == "human":
            self.losses += 1
        else:
            self.ties += 1

        trust_now = self._trust_scores()
        self.trust_log.append(dict(zip([p.name for p in self.predictors], trust_now)))

        self.round_log.append(
            {
                "round": len(self.history),
                "human": human_move,
                "agent": agent_move,
                "result": result,
                "win_rate": self.wins / len(self.history),
                "trusted_predictor": self.predictors[self._last_action].name,
                "confidence": trust_now[self._last_action],
            }
        )
        self._decay_epsilon()
        return result

    # ---------- introspection for the GUI ----------

    def _trust_scores(self):
        return [sum(w) / len(w) if w else 0.0 for w in self.accuracy_window]

    def status(self):
        trust = self._trust_scores()
        best_i = max(range(self.n), key=lambda i: trust[i]) if self.history else None
        return {
            "trust_scores": list(zip([p.name for p in self.predictors], trust)),
            "trusted_predictor": self.predictors[best_i].name if best_i is not None else None,
            "confidence": trust[best_i] if best_i is not None else 0.0,
            "rounds_played": len(self.history),
            "win_rate": self.wins / len(self.history) if self.history else 0.0,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
        }

    def q_table_frame(self):
        """Return (state_labels, predictor_names, matrix) for a heatmap of every
        visited state's learned Q-values -- literally what the agent has learned
        about which predictor to trust, and when."""
        names = [p.name for p in self.predictors]
        if not self.Q:
            return [], names, []
        states = sorted(self.Q.keys(), key=lambda s: (s.count("na"), s))
        matrix = [self.Q[s] for s in states]
        labels = [", ".join(s) for s in states]
        return labels, names, matrix

    def narration(self):
        if len(self.history) < 3:
            return "Not enough rounds yet — still calibrating to how you play."

        moves = [h["human"] for h in self.history]
        stay_after_win = win_rounds = shift_after_loss = loss_rounds = 0
        for i in range(1, len(self.history)):
            prev, curr = self.history[i - 1], self.history[i]
            if prev["result"] == "human":
                win_rounds += 1
                stay_after_win += curr["human"] == prev["human"]
            elif prev["result"] == "agent":
                loss_rounds += 1
                shift_after_loss += curr["human"] != prev["human"]

        trust = dict(zip([p.name for p in self.predictors], self._trust_scores()))
        leader = max(trust, key=trust.get)
        if leader == "Win-Stay / Lose-Shift" and (win_rounds >= 2 or loss_rounds >= 2):
            stay_pct = int(100 * stay_after_win / win_rounds) if win_rounds else 0
            shift_pct = int(100 * shift_after_loss / loss_rounds) if loss_rounds else 0
            return (f"You repeat your winning move {stay_pct}% of the time, and switch "
                    f"away from a losing move {shift_pct}% of the time. That's not random "
                    f"— that's a habit, and I'm using it.")
        if leader == "Frequency":
            counts = {m: moves.count(m) for m in MOVES}
            fav = max(counts, key=counts.get)
            return f"You favor {fav} — {int(100 * counts[fav] / len(moves))}% of your throws so far. I'm leaning on that."
        if leader == "Markov (last-move)" and len(moves) >= 2:
            return (f"Your last move was {moves[-1]}, and I've noticed what you tend to throw "
                    "right after that specific move. Predicting from that pattern now.")
        if leader == "Markov (last-2)" and len(moves) >= 3:
            return (f"Your last two moves were {moves[-2]} then {moves[-1]} — I've picked up "
                    "what you tend to do after that exact two-move sequence.")
        if leader == "Anti-Repeat":
            return "You're actively avoiding repeats — trying to 'look' random. That's a pattern too, and it's the one I'm exploiting right now."
        return "Reading your recent rhythm — no single habit dominates yet, blending signals."

    # ---------- named opponent profiles ----------

    def to_dict(self):
        """Return JSON-safe state so a named opponent can be resumed later."""
        predictor_state = []
        for predictor in self.predictors:
            predictor_state.append({
                "name": predictor.name,
                "transitions": getattr(predictor, "transitions", None),
            })
        return {
            "version": 2,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_start": self.epsilon_start,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "accuracy_window": self.accuracy_window,
            "q_table": {"|".join(state): values for state, values in self.Q.items()},
            "history": self.history,
            "round_log": self.round_log,
            "trust_log": self.trust_log,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "predictor_state": predictor_state,
        }

    @classmethod
    def from_dict(cls, data):
        agent = cls(
            alpha=data.get("alpha", 0.3),
            gamma=data.get("gamma", 0.7),
            epsilon=data.get("epsilon", 0.15),
            epsilon_start=data.get("epsilon_start"),
            epsilon_min=data.get("epsilon_min", 0.05),
            epsilon_decay=data.get("epsilon_decay", 1.0),
        )
        agent.accuracy_window = data.get("accuracy_window", [[] for _ in agent.predictors])
        agent.Q = defaultdict(lambda: [0.0] * agent.n)
        for key, values in data.get("q_table", {}).items():
            agent.Q[tuple(key.split("|"))] = values
        agent.history = data.get("history", [])
        agent.round_log = data.get("round_log", [])
        agent.trust_log = data.get("trust_log", [])
        agent.wins = data.get("wins", 0)
        agent.losses = data.get("losses", 0)
        agent.ties = data.get("ties", 0)
        for predictor, saved in zip(agent.predictors, data.get("predictor_state", [])):
            if saved.get("transitions") is not None and hasattr(predictor, "transitions"):
                predictor.transitions = saved["transitions"]
        return agent

    def save_profile(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, indent=2)

    @classmethod
    def load_profile(cls, path):
        with open(path, encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))


class UCB1BanditAgent(MirrorAgent):
    """Contextual UCB1 bandit over the same predictor actions as MirrorAgent.

    It learns the average immediate reward for each predictor within each
    discretized accuracy context, but has no Q-learning bootstrapping or
    discount factor. This makes it a direct algorithm comparison.
    """

    def __init__(self, c=1.4, seed=None):
        super().__init__(alpha=0.0, gamma=0.0, epsilon=0.0, seed=seed)
        self.ucb_c = c
        self.bandit_counts = defaultdict(lambda: [0] * self.n)
        self.bandit_values = defaultdict(lambda: [0.0] * self.n)

    def _choose_action(self, state):
        counts = self.bandit_counts[state]
        values = self.bandit_values[state]
        for action, count in enumerate(counts):
            if count == 0:
                return action
        total = sum(counts)
        scores = [
            values[i] + self.ucb_c * ((_log(total) / counts[i]) ** 0.5)
            for i in range(self.n)
        ]
        best = max(scores)
        return random.choice([i for i, score in enumerate(scores) if score == best])

    def learn(self, human_move, agent_move):
        result = super().learn(human_move, agent_move)
        state = self._last_state
        action = self._last_action
        reward = {"agent": 1.0, "tie": -0.2, "human": -1.0}[result]
        counts = self.bandit_counts[state]
        values = self.bandit_values[state]
        counts[action] += 1
        values[action] += (reward - values[action]) / counts[action]
        return result


def _log(value):
    """Small local log helper keeps agent.py dependency-free."""
    import math
    return math.log(max(value, 1))
