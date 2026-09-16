"""
simulate_players.py
--------------------
Simulated "human" archetypes used for offline evaluation (evaluate.py) since
we can't put thousands of real humans through the live GUI for a report.
Each archetype plays with a different, realistic behavioral bias, mirroring
well-documented human RPS tendencies from behavioral-game-theory literature.

Every function has the same signature: given the game history so far
(list of {"human","agent","result"} dicts, from the PLAYER's point of view),
return the player's next move.
"""

import random

MOVES = ["rock", "paper", "scissors"]


def random_player(history):
    """A theoretically unbeatable player: uniform random every round."""
    return random.choice(MOVES)


def rock_biased_player(history):
    """Real humans over-play Rock as an opening/default move. Simple frequency bias."""
    return random.choices(MOVES, weights=[0.5, 0.25, 0.25])[0]


def win_stay_lose_shift_player(history):
    """Repeats winning moves, switches after losing — one of the most common
    documented human biases in repeated RPS play."""
    if not history:
        return random.choice(MOVES)
    last = history[-1]
    if last["result"] == "human":
        return last["human"] if random.random() < 0.75 else random.choice(MOVES)
    elif last["result"] == "agent":
        others = [m for m in MOVES if m != last["human"]]
        return random.choice(others) if random.random() < 0.7 else last["human"]
    else:
        return random.choice(MOVES)


def cyclic_player(history):
    """Cycles rock -> paper -> scissors -> rock ..., a very common
    'trying to look random but actually predictable' human pattern."""
    order = ["rock", "paper", "scissors"]
    if not history:
        return random.choice(MOVES)
    last = history[-1]["human"]
    idx = order.index(last)
    nxt = order[(idx + 1) % 3]
    return nxt if random.random() < 0.65 else random.choice(MOVES)


def anti_repeat_player(history):
    """Deliberately avoids repeating their last move (over-corrected 'randomness')."""
    if not history:
        return random.choice(MOVES)
    last = history[-1]["human"]
    others = [m for m in MOVES if m != last]
    return random.choice(others) if random.random() < 0.8 else last


ARCHETYPES = {
    "random": random_player,
    "rock_biased": rock_biased_player,
    "win_stay_lose_shift": win_stay_lose_shift_player,
    "cyclic": cyclic_player,
    "anti_repeat": anti_repeat_player,
}
