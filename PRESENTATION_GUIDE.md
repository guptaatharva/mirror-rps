# Presenting "The Mirror" — Viva / Demo Script

A walkthrough for explaining this project out loud: what to say, what to click, and
what to have ready when someone pushes back with a question.

---

## 1. The 30-second pitch

> "Most people asked to play Rock-Paper-Scissors 'randomly' can't actually do it —
> they repeat winning moves, switch after losing, or fall into cycles when they try
> too hard to look unpredictable. **The Mirror** is a reinforcement-learning agent
> that never gets told this in advance — it plays you live, tries four or five
> different theories about your habits simultaneously, and learns through pure
> win/loss reward which theory is actually working on *you*. It then explains, in
> plain English, what it thinks your tell is."

That sentence alone answers "what is this" — everything else in the demo is proof.

---

## 2. The problem statement (say this before touching the keyboard)

- **Domain:** behavioral game theory / human predictability under RPS.
- **Real-world hook:** the same "humans aren't as random as they think" phenomenon
  shows up in security (password choices), UX (dark-pattern resistance), and
  game AI — RPS is just the cleanest, fastest environment to demonstrate it in
  live, in front of an audience, in under a minute per round.
- **Objective:** an agent that (a) starts at the 33% chance floor against a brand
  new opponent, (b) climbs measurably above it *live*, during the presentation,
  against a real person in the room, and (c) can say *why*.
- **Why this counts as "no labeled dataset":** there's no CSV of pre-recorded RPS
  games. The "training data" is generated live by whoever is playing — that's the
  environment, not a dataset, which is exactly what an RL project is supposed to
  look like.

---

## 3. Why Reinforcement Learning, specifically (pre-empt the "why not classification" question)

Say this explicitly — it shows you understood the choice, not just implemented it:

- There is no fixed label to predict in advance ("this is a rock-thrower") — the
  right answer changes per opponent and even shifts mid-game if they get self-aware.
- The agent has to *act* (pick a move) and only finds out if it was right after
  committing — that's an action → reward loop, not a static prediction task.
- We're not learning "what beats rock" (trivial, hardcoded) — we're learning
  **which of several competing theories about the opponent to trust right now**,
  purely from reward. That's a policy over meta-actions, which is the RL framing.

---

## 4. System architecture — the 90-second technical walkthrough

Draw this on a whiteboard or just say it in order:

```
human's move history
        │
        ▼
┌───────────────────────────────────────────────┐
│  5 behavioral predictors, each guesses your    │
│  NEXT move from a different theory:            │
│   • Frequency          — you favor one move    │
│   • Markov (last-1)    — depends on last move  │
│   • Markov (last-2)    — depends on last 2     │
│   • Outcome Reaction   — learns YOUR stay/shift│
│                          tendency after a win  │
│                          vs. after a loss, sep-│
│                          arately (not assumed) │
│   • Anti-Repeat        — you avoid repeats     │
└───────────────────────────────────────────────┘
        │  each predictor's last-10-round accuracy
        ▼  is discretized into na/low/med/high
┌───────────────────────────────────────────────┐
│  Q-learning meta-agent (MirrorAgent)           │
│  STATE  = accuracy bucket of all 5 predictors  │
│  ACTION = which predictor to trust this round  │
│  REWARD = +1 win / −0.2 tie / −1 loss          │
└───────────────────────────────────────────────┘
        │
        ▼
  counter the trusted predictor's guess → play that move
```

Key line to say out loud: **"The Q-learning layer is never told which predictor is
best. It discovers that live, per opponent, purely from whether it won or lost the
last round."** That single sentence is the entire RL justification — memorize it.

---

## 5. Dimensionality reduction — don't skip this, it's graded separately

- Every played session (yours, or a simulated archetype) is reduced to a
  **6-dimensional behavioral fingerprint**: 3 move frequencies + stay-after-win
  rate + shift-after-loss rate + repeat rate.
- We standardize those 6 features and run **PCA → 2 components**.
- The payoff: the resulting scatter plot (`results/pca_player_types.png`, or live
  in the **Simulation Lab** tab) shows the 6 simulated archetypes occupying
  **visibly separate regions** of that 2D space. That's the justification — PCA
  isn't decoration here, it's proof that "play style" is a real, measurable,
  separable thing in this feature space, not just a story we're telling.
- If asked "why only 2 components": say variance explained is reported directly
  on the chart's axis labels (PC1/PC2 %), and 2D is chosen specifically because
  it needs to be human-readable on a slide/screen, not because more components
  wouldn't work.

---

## 6. Evaluation results — the numbers to actually quote

From `results/comparison_table.csv` and `results/significance_summary.csv`
(200 rounds × 15 trials per cell):

| Opponent archetype | Mirror's win rate | vs. 33% baseline | Statistically significant? |
|---|---|---|---|
| Random (true baseline) | 33.1% | flat | No — **correctly** flat (p ≈ 0.15) |
| Rock-biased | 41.4% | +8.4 pts | Yes, 11/15 trials (p ≈ 1.2×10⁻⁵) |
| Win-stay / lose-shift | 40.7% | +7.7 pts | Yes, 10/15 trials (p ≈ 0.0007) |
| Lose-stay / win-shift (opposite bias) | 51.9% | **+18.9 pts** | Yes, 14/15 trials (p ≈ 2.7×10⁻²²) |
| Cyclic | 63.2% | **+30.2 pts** | Yes, 15/15 trials (p ≈ 3.6×10⁻²⁴) |
| Anti-repeat | 38.6% | +5.6 pts | Yes, 5/15 trials (p ≈ 0.001) |

**Say this explicitly, don't wait for the question:** *"Against a truly random
player, we stay at 33% — that's correct, not a failure. Information theory
guarantees no model beats uniform randomness, and showing we don't fake a win
there is itself evidence the system isn't cheating."*

**The `lose_stay_win_shift` row is your best answer to "what if someone doesn't
fit the textbook pattern?"** It's a simulated player who does the *opposite* of
win-stay/lose-shift — repeats after losing, switches after winning. A predictor
hard-coded to assume the textbook direction would actively work against this
player. Ours doesn't hard-code a direction: the `Outcome Reaction` predictor
learns stay-after-win and stay-after-loss rates separately from *this specific
opponent's* history, so it catches this reversed pattern just as well —
`Outcome Reaction` alone hits 64.8% here, almost identical to its performance
on the textbook direction.

**Algorithm comparison** (same table): a single fixed Markov predictor actually
*beats* the full Q-learning ensemble on the cyclic archetype (75.8% vs. 63.2%).
This is a good, honest talking point, not a flaw to hide:

> "A specialist that's hard-coded to expect exactly this pattern will always
> beat a generalist that has to spend some of its rounds exploring (ε-greedy)
> and hasn't committed early. The ensemble's value is that it doesn't need to
> be told in advance which specialist to be — it still beats the 33% floor by
> a wide margin on every non-random archetype, without that prior knowledge."

We also ran a **UCB1 bandit** as a second, simpler learning algorithm for direct
comparison (no Q-learning bootstrapping) — it tracks Q-learning closely but
without the discounted-future term, which is exactly the ablation you'd want to
show "the Q-learning-specific part is pulling weight, not just 'any learning.'"

---

## 7. Live demo script — what to actually click, in order

1. **Play tab** — pick a *deliberate* pattern before you start (e.g. "I'll cycle
   rock→paper→scissors" or "I'll always repeat after I win"). Play ~15–20 rounds
   live. Narrate what you're doing to the room as you play, so they can verify
   the agent's narration is genuinely tracking you and not generic filler.
   Point at the "What I've learned about you" panel and the win-rate chart
   crossing above the 33% dashed line.
2. **Live Mind Model tab** — show the Q-table heatmap: *"this is what it
   actually learned — every row is a state it visited, every column a predictor,
   the color is how much it trusts that predictor in that state."* Then the
   trust-over-time chart: point at the line for whatever predictor matches the
   pattern you played and show it climbing while the others stay near 33%.
3. **Simulation Lab tab** — pick the `cyclic` archetype vs. `Mirror`, 100 rounds,
   5 trials, hit run. This is your best number (60%+ win rate) and it takes
   under 2 seconds to generate live in front of the room — much stronger than a
   static screenshot. Then re-run against `random` to show it correctly stays
   near 33% there too. Scroll to the PCA plot and point out the ★ marker landing
   inside the correct archetype's cluster.
4. **About tab** — only if asked for the report figures / rubric mapping; this
   is your fallback/reference tab, not a headline demo step.

---

## 8. Rubric → where to point when asked "where does X requirement show up"

| Rubric item | Point to |
|---|---|
| Problem definition | §2 above / README §1 |
| Dataset/environment + EDA | No static CSV — environment is live/simulated play; EDA = archetype behavioral-feature distributions, `results/behavioral_features.csv`, visible live in Simulation Lab |
| Algorithm implementation | `agent.py` — Q-learning update rule, live in Live Mind Model tab |
| Dimensionality reduction | `analysis.py` PCA, §5 above |
| Evaluation metrics | Win rate, exact binomial significance tests, `evaluate.py` |
| Algorithm comparison | Mirror vs UCB1 vs Random vs 5 single predictors, §6 above |
| Real-time GUI | The whole app — nothing is pre-computed, every tab runs live |

---

## 9. Anticipated questions (viva prep)

**"Why not deep Q-learning / neural nets?"**
> The state space is tiny by design (4 buckets ^ 5 predictors, and only a few
> hundred rounds of data per opponent at most) — a tabular Q-table converges
> faster and is fully interpretable (we can literally show the whole thing in a
> heatmap), which matters more here than raw capacity a neural net would need
> far more data to use well.

**"How do you know it's not overfitting to your specific test archetypes?"**
> The predictors encode general, well-documented human RPS biases from
> behavioral-game-theory literature (frequency bias, win-stay/lose-shift,
> Markov dependence, anti-repeat over-correction), not anything tuned to our
> six simulated archetypes specifically. The archetypes exist to *validate*
> that the predictors work, not to define them.

**"What happens against a perfectly random human?"**
> Win rate stays at 33% (shown live in Simulation Lab, and in the table above)
> — this is mathematically the best any method can do against true uniform
> randomness, so a flat 33% there is confirmation the system is honest, not a
> weakness.

**"Why discretize the state into buckets instead of using raw accuracy?"**
> A continuous state space would need function approximation (e.g. a neural
> net) to generalize across states never seen exactly before — a small number
> of discrete buckets lets a plain Q-table converge within the ~15–40 rounds a
> live demo actually has time for.

**"What's the actual reward signal, precisely?"**
> +1 for a round the agent wins, −1 for a round it loses, −0.2 for a tie (ties
> are mildly discouraged so the agent keeps hunting for an edge rather than
> settling). See `agent.py`'s `learn()` method.

**"Could this generalize beyond RPS?"**
> Yes — the pattern (ensemble of cheap behavioral heuristics + a meta-learner
> that picks which one to trust, live, per-individual) is the same shape as
> real recommendation-strategy and adaptive-difficulty problems; RPS is chosen
> here because it's fast enough to demonstrate convergence live, in front of an
> audience, in under a minute.

---

## 10. Closing line

> "The headline number isn't any single win rate — it's that the same untouched
> code, with zero prior knowledge of the opponent, correctly does three different
> things depending on who's playing: stays at chance against true randomness,
> climbs steadily against realistic human habits, and can tell you in plain
> language which habit it's exploiting, all live, all in this room."
