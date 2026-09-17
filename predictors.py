"""
predictors.py
-------------
Core game logic + the ensemble of "behavioral" predictors that try to guess
the human's NEXT move from their move history. Each predictor encodes a
different psychological tell that real humans exhibit when they think they
are being random:

  - FrequencyPredictor        : people over-favor one move overall (Rock bias is common)
  - MarkovPredictor           : people's next move depends on their last move
  - Markov2Predictor          : people's next move depends on their last TWO moves
  - OutcomeReactionPredictor  : people react to winning/losing by staying or shifting --
                                learned per-outcome, since not everyone is the classic
                                win-stay/lose-shift direction (some are the opposite)
  - AntiPatternPredictor      : people deliberately avoid repeats when "trying to be random"

The Q-learning meta-agent in agent.py learns, live, which of these predictors
to trust for a given opponent.
"""

import random

MOVES = ["rock", "paper", "scissors"]

# BEATS[x] = the move that beats x
BEATS = {"rock": "paper", "paper": "scissors", "scissors": "rock"}


def counter_move(predicted_human_move: str) -> str:
    """Return the move that beats the predicted human move."""
    return BEATS[predicted_human_move]


def outcome(human_move: str, agent_move: str) -> str:
    """Return 'human', 'agent', or 'tie' from the agent's perspective."""
    if human_move == agent_move:
        return "tie"
    if BEATS[human_move] == agent_move:
        return "agent"  # agent's move beats human's move
    return "human"  # human's move beats agent's move


class BasePredictor:
    name = "base"

    def update(self, history):
        """Optional: called every round with the full history so far."""
        pass

    def predict(self, history):
        """Return a predicted NEXT human move given the round history."""
        return random.choice(MOVES)


class FrequencyPredictor(BasePredictor):
    """Bets the human will repeat whatever move they play most often overall."""
    name = "Frequency"

    def predict(self, history):
        moves = [h["human"] for h in history]
        if not moves:
            return random.choice(MOVES)
        counts = {m: moves.count(m) for m in MOVES}
        top = max(counts.values())
        best = [m for m, c in counts.items() if c == top]
        return random.choice(best)


class MarkovPredictor(BasePredictor):
    """Bets on P(next move | last move) — a first-order Markov chain of the human's play."""
    name = "Markov (last-move)"

    def __init__(self):
        self.transitions = {m: {m2: 0 for m2 in MOVES} for m in MOVES}

    def update(self, history):
        if len(history) >= 2:
            prev = history[-2]["human"]
            curr = history[-1]["human"]
            self.transitions[prev][curr] += 1

    def predict(self, history):
        if not history:
            return random.choice(MOVES)
        last = history[-1]["human"]
        row = self.transitions[last]
        if sum(row.values()) == 0:
            return random.choice(MOVES)
        top = max(row.values())
        best = [m for m, c in row.items() if c == top]
        return random.choice(best)


class Markov2Predictor(BasePredictor):
    """Bets on P(next move | last two moves) -- a second-order Markov chain.
    Catches longer, more deliberate patterns (e.g. rock-rock-paper habits)
    that the first-order Markov predictor is too short-sighted to see."""
    name = "Markov (last-2)"

    def __init__(self):
        self.transitions = {}
        for a in MOVES:
            for b in MOVES:
                self.transitions[f"{a},{b}"] = {m: 0 for m in MOVES}

    def update(self, history):
        if len(history) >= 3:
            key = f"{history[-3]['human']},{history[-2]['human']}"
            curr = history[-1]["human"]
            self.transitions[key][curr] += 1

    def predict(self, history):
        if len(history) < 2:
            return random.choice(MOVES)
        key = f"{history[-2]['human']},{history[-1]['human']}"
        row = self.transitions[key]
        if sum(row.values()) == 0:
            return random.choice(MOVES)
        top = max(row.values())
        best = [m for m, c in row.items() if c == top]
        return random.choice(best)


class OutcomeReactionPredictor(BasePredictor):
    """Generalizes the classic 'win-stay/lose-shift' tell instead of assuming
    its direction. Real players don't all react the same way to winning and
    losing: some repeat after winning and switch after losing (textbook
    win-stay/lose-shift); others get stubborn and repeat after LOSING, on a
    gambler's-fallacy "it'll work this time" instinct; others switch even
    after winning, trying to look unpredictable. This predictor tracks, per
    outcome (win/loss/tie), how often THIS player actually stays vs. shifts,
    and bets on whichever tendency it has actually observed -- so it catches
    any of these reaction styles, not just the textbook one."""
    name = "Outcome Reaction"

    def __init__(self):
        self.counts = {r: {"stay": 0, "shift": 0} for r in ("human", "agent", "tie")}

    def update(self, history):
        if len(history) >= 2:
            prev, curr = history[-2], history[-1]
            stayed = curr["human"] == prev["human"]
            self.counts[prev["result"]]["stay" if stayed else "shift"] += 1

    def predict(self, history):
        if not history:
            return random.choice(MOVES)
        last = history[-1]
        c = self.counts[last["result"]]
        total = c["stay"] + c["shift"]
        if total == 0:
            return random.choice(MOVES)
        if c["stay"] / total >= 0.5:
            return last["human"]
        others = [m for m in MOVES if m != last["human"]]
        return random.choice(others)


class AntiPatternPredictor(BasePredictor):
    """Catches people who are TRYING to be unpredictable: they avoid playing the
    same move 3x in a row, and tend to cycle through all three moves quickly."""
    name = "Anti-Repeat"

    def predict(self, history):
        if len(history) < 2:
            return random.choice(MOVES)
        last_two = [history[-1]["human"], history[-2]["human"]]
        if last_two[0] == last_two[1]:
            # they just repeated once -> unlikely to make it three in a row
            others = [m for m in MOVES if m != last_two[0]]
            return random.choice(others)
        played_recently = set(last_two)
        remaining = [m for m in MOVES if m not in played_recently]
        if remaining:
            return remaining[0]  # they've been cycling -> bet on the "missing" move
        return random.choice(MOVES)


def build_predictor_bank():
    """Factory so app.py / evaluate.py always build predictors in the same order."""
    return [
        FrequencyPredictor(),
        MarkovPredictor(),
        Markov2Predictor(),
        OutcomeReactionPredictor(),
        AntiPatternPredictor(),
    ]
