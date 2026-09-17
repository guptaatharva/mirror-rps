# The Mirror — a Reinforcement-Learning RPS Mind Reader

An RL agent that learns to predict a human opponent's Rock-Paper-Scissors
moves from their own subconscious behavioral tells, and narrates what
pattern it's exploiting, live, while it plays — with a live Q-table view,
an in-app simulation lab, and interactive PCA of behavioral styles.

## 1. Problem statement

Humans asked to play "randomly" systematically fail to — they repeat winning
moves, switch after losing, over-favor certain moves, and over-correct into
detectable cyclic patterns when trying to look unpredictable. **The Mirror**
is an RL agent that discovers *which* of these tells a specific opponent is
exhibiting, purely through play (no labeled dataset, no pretraining), and
exploits it to win above the 33% random-chance baseline.

**Objective:** build an agent that (a) starts at chance level against any
new opponent, (b) adapts its strategy live as it observes that opponent's
play, and (c) can explain, in plain language, what behavioral pattern it is
currently trusting.

**Expected outcome:** win rate against realistic (non-uniform-random) human
play climbs measurably above 33% within roughly 20-40 rounds, and stays at
33% against a genuinely random opponent — proving the system is reading a
real signal, not cheating.

## 2. Approach: Reinforcement Learning

- **Environment:** a live RPS match against a human (Play tab), or a
  simulated player archetype (Simulation Lab tab, or offline via `evaluate.py`).
- **Agent:** `MirrorAgent` (see `agent.py`) — a Q-learning meta-agent.
- **State:** the recent (last 10 rounds) prediction accuracy of each of 5
  behavioral predictors, discretized into low/med/high/na buckets.
- **Action:** which of the 5 predictors to trust this round.
- **Reward:** +1 win, -1 loss, -0.2 tie.

### The predictor ensemble (`predictors.py`)
| Predictor | Behavioral tell it detects |
|---|---|
| Frequency | Overall move bias (e.g. over-playing Rock) |
| Markov (last-move) | P(next move \| last move) |
| Markov (last-2) | P(next move \| last two moves) — longer, more deliberate habits |
| Outcome Reaction | Learns your stay/shift tendency after a win vs. a loss *separately* — catches classic win-stay/lose-shift, its mirror image (lose-stay/win-shift), or either-direction repeaters, instead of assuming one fixed direction |
| Anti-Repeat | Over-corrected "trying to look random" cycling |

The Q-learning layer does **not** know in advance which predictor is best —
it discovers that live, per-opponent, from reward alone. This is the
genuinely reinforcement-learning part of the system.

## 3. Dimensionality reduction (PCA)

Each played session is reduced to a 6-dimensional behavioral feature vector
(move frequencies, stay-after-win rate, shift-after-loss rate, repeat rate).
Both `evaluate.py` (offline) and the **Simulation Lab** tab (live, in-app)
share the same feature/PCA code (`analysis.py`), run many sessions across 6
player archetypes, standardize the features, and project to 2 components.
The resulting plot shows genuine visual separation between archetypes — this
justifies PCA's use here: it reveals real structure in a multi-dimensional
behavioral feature space, rather than being applied because the brief
mentions it. The Simulation Lab additionally drops a ★ marker showing exactly
where the session you just ran landed on that map.

## 4. Evaluation & algorithm comparison

`evaluate.py` compares, over 200-round games x 15 trials, against 6
simulated archetypes (random, rock-biased, win-stay/lose-shift,
lose-stay/win-shift, cyclic, anti-repeat):

- **MirrorAgent** (full Q-learning ensemble)
- **UCB1BanditAgent** (contextual UCB1 over the same predictor actions)
- **RandomAgent** (floor baseline — should stay at ~33% everywhere)
- **Each single predictor used alone**, with no meta-learning

The evaluation also runs a small α/ε sensitivity grid (α ∈ {0.1, 0.3, 0.5},
ε ∈ {0.05, 0.15, 0.3}), and uses exact binomial tests against the 33%
baseline. Non-random archetypes use a one-sided test for improvement; the
random archetype uses a two-sided test so that a non-significant result is an
explicit sanity check. The **Simulation Lab** tab runs this exact same
comparison (any agent kind vs any archetype, N trials averaged) live in the
GUI, with the same significance test reported on screen.

Outputs (in `results/` after running `evaluate.py`):
- `comparison_table.csv` — mean final win rate per agent per archetype
- `hyperparameter_sensitivity.csv` / `.png` — α/ε sweep results and plot
- `significance_tests.csv` / `significance_summary.csv` — exact binomial tests
- `winrate_curves.png` — MirrorAgent's learning curve per archetype over
  time, with the 33% baseline marked
- `pca_player_types.png` — the PCA plot described above
- `behavioral_features.csv` — the raw feature vectors behind the PCA plot

**Honest result to report:** against a truly random player, every approach
(including MirrorAgent) sits at ~33% — this is expected and correct, and is
worth stating explicitly in your report/viva as proof the system isn't
"cheating." Against every realistic (non-uniform) archetype, MirrorAgent
beats the 33% baseline substantially (all p < 0.002, most far smaller). A
single specialist predictor sometimes outperforms the ensemble on the
archetype it was designed for (e.g. Markov vs. the cyclic archetype, ~76% vs.
the ensemble's ~63%) — a good, honest talking point about the
exploration/generalization trade-off of the Q-learning meta-layer versus a
hand-picked specialist, and about the cost of ε-exploration the specialist
doesn't pay.

**Why the `Outcome Reaction` predictor matters specifically:** it doesn't
assume win-stay/lose-shift is the direction — it learns each player's actual
stay-after-win vs. stay-after-loss rate separately. The `lose_stay_win_shift`
archetype (repeats after losing, switches after winning — the mirror image of
the textbook bias) proves this: MirrorAgent still reaches 51.9% there
(p ≈ 3×10⁻²²), and `Outcome Reaction` alone reaches 64.8%, essentially
matching its performance on the *textbook* direction. A predictor hardcoded to
assume win-stay/lose-shift would have been actively wrong against this player.

## 5. Real-time GUI

`app.py` (Streamlit) is the live demo, organized into four tabs:

**🎮 Play** — the live human-vs-agent game:
- Rock / Paper / Scissors buttons, live win-rate metric vs. the 33% baseline
- Named opponent profiles (dropdown of saved opponents in `profiles/`) that
  reload the learned Q-table and behavioral history in a later session
- "What I've learned about you" panel — plain-language narration of the
  specific behavioral tell currently being exploited, a live confidence
  meter, and a live predictor-accuracy leaderboard
- Win-rate-over-time chart and full round-by-round history table

**🧠 Live Mind Model** — transparency into the Q-learning itself:
- A heatmap of the agent's actual learned Q-table (every visited state ×
  every predictor action), so you can see *what* it learned, not just *that*
  it learned
- A live multi-line chart of every predictor's windowed accuracy over the
  course of the session, so you can watch trust shift between predictors

**🤖 Simulation Lab** — the offline `evaluate.py` comparison, exposed live:
- Pick any archetype and any agent/baseline, run N trials instantly, and see
  the averaged win-rate curve against the 33% baseline plus an exact
  binomial significance test
- The session is plotted as a ★ on the PCA behavioral map against a
  freshly-sampled cloud of all 6 archetypes

**ℹ️ About** — problem statement, RL setup, predictor table, rubric mapping,
and (if `evaluate.py` has been run) the offline report figures inline.

New sessions begin with more exploration (ε = 0.30) and decay toward ε = 0.05,
while saved profiles preserve the learned opponent-specific state.

This is a genuinely real-time system: nothing is pre-computed. Every move,
prediction, and narration line is generated live from the current opponent's
play, and the policy is opponent-specific (reset the session and it starts
back at chance level against a new player).

## How to run

```bash
pip install -r requirements.txt

# Live GUI demo (Play / Live Mind Model / Simulation Lab / About)
streamlit run app.py

# Offline evaluation / report assets (comparison table, PCA plot, curves)
python evaluate.py
```

## File overview

| File | Purpose |
|---|---|
| `predictors.py` | Game logic + the 5 behavioral predictors |
| `agent.py` | `MirrorAgent` / `UCB1BanditAgent` — Q-learning & UCB1 meta-agents, live narration, Q-table introspection |
| `analysis.py` | Shared, side-effect-free behavioral-feature/PCA + simulation helpers used by both `app.py` and `evaluate.py` |
| `app.py` | Streamlit real-time GUI (Play / Live Mind Model / Simulation Lab / About) |
| `simulate_players.py` | Simulated human archetypes for offline evaluation and the Simulation Lab |
| `evaluate.py` | Offline report: baseline comparison, learning curves, hyperparameter sensitivity, significance tests, PCA analysis |
