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
  - WinStayLoseShiftPredictor : people repeat after winning, switch after losing
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


class WinStayLoseShiftPredictor(BasePredictor):
    """Classic behavioral-economics tell: people repeat winning moves, and switch
    away from moves that just lost, more often than pure chance would predict."""
    name = "Win-Stay / Lose-Shift"

    def predict(self, history):
        if not history:
            return random.choice(MOVES)
        last = history[-1]
        if last["result"] == "human":  # the human just won -> likely to stay
            return last["human"]
        elif last["result"] == "agent":  # the human just lost -> likely to shift
            others = [m for m in MOVES if m != last["human"]]
            return random.choice(others)
        else:  # tie -> mild tendency to switch
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
        WinStayLoseShiftPredictor(),
        AntiPatternPredictor(),
    ]
