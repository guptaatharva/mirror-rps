"""
app.py
------
"THE MIRROR" — a live Streamlit demo of an RL agent that learns to predict a
human opponent's Rock-Paper-Scissors moves from their own behavioral tells,
and narrates what it's picking up on in plain language while it plays.

Tabs:
  Play              - the live human-vs-agent game.
  Live Mind Model   - what the Q-learning agent has actually learned this
                       session: the Q-table heatmap and per-predictor trust
                       over time.
  Simulation Lab     - run the agent (or a baseline) against simulated human
                       archetypes instantly, right inside the GUI, and see
                       the session plotted against the PCA behavioral map.
  About              - problem statement, methodology, rubric mapping.

Run with:  streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from agent import MirrorAgent
from predictors import build_predictor_bank
from simulate_players import ARCHETYPES
from analysis import behavioral_features, fit_pca, run_game, exact_binomial_pvalue

PROFILE_DIR = Path("profiles")
MOVE_EMOJI = {"rock": "\U0001FAA8", "paper": "\U0001F4C4", "scissors": "✂️"}
ACCENT = "#7C5CFC"
WIN_COLOR = "#22c55e"
LOSS_COLOR = "#ef4444"
TIE_COLOR = "#94a3b8"

SIM_AGENT_KINDS = {
    "Mirror (Q-learning ensemble)": "mirror",
    "UCB1 bandit (no bootstrapping)": "ucb1",
    "Random baseline (no learning)": "random",
}
for _p in build_predictor_bank():
    SIM_AGENT_KINDS[f"Single predictor: {_p.name}"] = _p.name


# =====================================================================
# helpers
# =====================================================================

def profile_path(name: str) -> Path:
    safe_name = "".join(c for c in name if c.isalnum() or c in "-_ ").strip()
    return PROFILE_DIR / f"{safe_name}.json"


def list_profiles():
    if not PROFILE_DIR.exists():
        return []
    return sorted(p.stem for p in PROFILE_DIR.glob("*.json"))


def new_session_agent():
    return MirrorAgent(epsilon=0.30, epsilon_min=0.05, epsilon_decay=0.985)


def inject_css():
    st.markdown(
        f"""
        <style>
        .stApp {{ background: radial-gradient(circle at top left, #1a1530 0%, #0f0c1a 45%, #0b0912 100%); }}
        [data-testid="stHeader"] {{ background: transparent; }}
        .mirror-hero {{
            padding: 1.4rem 1.8rem; border-radius: 18px; margin-bottom: 1.2rem;
            background: linear-gradient(120deg, rgba(124,92,252,0.28), rgba(34,197,94,0.10));
            border: 1px solid rgba(124,92,252,0.35);
        }}
        .mirror-hero h1 {{ margin: 0 0 .3rem 0; font-size: 2rem; }}
        .mirror-hero p {{ margin: 0; opacity: .85; font-size: .95rem; }}
        div[data-testid="stMetric"] {{
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
            border-radius: 14px; padding: .6rem .9rem;
        }}
        .trust-card {{
            border-radius: 14px; padding: .9rem 1.1rem; margin-bottom: .6rem;
            background: rgba(124,92,252,0.10); border: 1px solid rgba(124,92,252,0.25);
        }}
        .reveal-card {{
            border-radius: 16px; padding: 1rem 1.2rem; text-align: center;
            background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08);
        }}
        .stButton>button {{ border-radius: 12px; font-weight: 600; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def plotly_layout(fig, height=360):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


st.set_page_config(page_title="The Mirror — RPS Mind Reader", page_icon="\U0001FAA8", layout="wide")
inject_css()

# ---------------- session state ----------------
if "agent" not in st.session_state:
    st.session_state.agent = MirrorAgent()
if "last_round" not in st.session_state:
    st.session_state.last_round = None
if "loaded_profile" not in st.session_state:
    st.session_state.loaded_profile = ""
if "sim_result" not in st.session_state:
    st.session_state.sim_result = None

# =====================================================================
# sidebar — opponent profile management
# =====================================================================
with st.sidebar:
    st.subheader("\U0001F464 Opponent profile")
    existing = list_profiles()
    options = ["— new opponent —"] + existing
    current_index = options.index(st.session_state.loaded_profile) if st.session_state.loaded_profile in existing else 0
    choice = st.selectbox("Load a saved opponent, or start new", options, index=current_index)

    if choice == "— new opponent —":
        profile_name = st.text_input("Name this opponent (optional, to save progress)", value="").strip()
    else:
        profile_name = choice

    if profile_name != st.session_state.loaded_profile:
        if profile_name:
            saved_path = profile_path(profile_name)
            if saved_path.name != ".json" and saved_path.exists():
                st.session_state.agent = MirrorAgent.load_profile(saved_path)
                st.success(f"Loaded {profile_name}'s profile — {st.session_state.agent.status()['rounds_played']} rounds of history.")
            else:
                st.session_state.agent = new_session_agent()
        else:
            st.session_state.agent = MirrorAgent()
        st.session_state.last_round = None
        st.session_state.loaded_profile = profile_name
        st.rerun()

    if profile_name:
        st.caption("Saved after every round.")

    st.write("---")
    agent: MirrorAgent = st.session_state.agent
    st.subheader("⚙️ Q-learning hyperparameters")
    st.caption(f"α (learning rate) = **{agent.alpha}**  \nγ (discount) = **{agent.gamma}**  \nε (exploration, live) = **{agent.epsilon:.3f}**")

    st.write("---")
    if st.button("\U0001F504 Reset this session", width="stretch"):
        st.session_state.agent = new_session_agent()
        st.session_state.last_round = None
        if profile_name:
            saved_path = profile_path(profile_name)
            if saved_path.name != ".json":
                st.session_state.agent.save_profile(saved_path)
        st.rerun()

agent: MirrorAgent = st.session_state.agent

# ---------------- header ----------------
st.markdown(
    """
    <div class="mirror-hero">
        <h1>\U0001FAA8 The Mirror</h1>
        <p>An RL agent that learns to read <i>how</i> you play — not what you type, what you throw.
        It doesn't win by cheating. It wins by noticing habits you don't know you have.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_play, tab_mind, tab_sim, tab_about = st.tabs(
    ["\U0001F3AE Play", "\U0001F9E0 Live Mind Model", "\U0001F916 Simulation Lab", "ℹ️ About"]
)

# =====================================================================
# TAB 1 — Play
# =====================================================================
with tab_play:
    col_game, col_mind = st.columns([1.15, 1])

    with col_game:
        status = agent.status()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Round", status["rounds_played"])
        m2.metric("Your win rate", f'{status["win_rate"]*100:.0f}%' if status["rounds_played"] else "—")
        m3.metric("Random-chance baseline", "33%")
        m4.metric("Agent record (W-L-T)", f'{status["wins"]}-{status["losses"]}-{status["ties"]}')

        st.write("")
        b1, b2, b3 = st.columns(3)
        human_choice = None
        if b1.button("\U0001FAA8 Rock", width="stretch"):
            human_choice = "rock"
        if b2.button("\U0001F4C4 Paper", width="stretch"):
            human_choice = "paper"
        if b3.button("✂️ Scissors", width="stretch"):
            human_choice = "scissors"

        if human_choice is not None:
            agent_move, predictions, action = agent.choose_move()
            result = agent.learn(human_choice, agent_move)
            st.session_state.last_round = {
                "human": human_choice,
                "agent": agent_move,
                "result": result,
                "trusted": agent.predictors[action].name,
            }
            if profile_name:
                saved_path = profile_path(profile_name)
                if saved_path.name != ".json":
                    agent.save_profile(saved_path)

        if st.session_state.last_round:
            r = st.session_state.last_round
            verdict_color = {"agent": LOSS_COLOR, "human": WIN_COLOR, "tie": TIE_COLOR}[r["result"]]
            verdict_text = {"agent": "\U0001F916 Mirror wins", "human": "\U0001F9CD You win", "tie": "\U0001F91D Tie"}[r["result"]]
            st.write("")
            rc1, rc2, rc3 = st.columns(3)
            rc1.markdown(f'<div class="reveal-card"><h3>You</h3><div style="font-size:2.2rem">{MOVE_EMOJI[r["human"]]}</div>{r["human"].title()}</div>', unsafe_allow_html=True)
            rc2.markdown(f'<div class="reveal-card"><h3>Mirror</h3><div style="font-size:2.2rem">{MOVE_EMOJI[r["agent"]]}</div>{r["agent"].title()}</div>', unsafe_allow_html=True)
            rc3.markdown(f'<div class="reveal-card" style="border-color:{verdict_color}"><h3>Result</h3><div style="font-size:1.3rem;color:{verdict_color}">{verdict_text}</div></div>', unsafe_allow_html=True)
            st.caption(f"Move chosen by trusting: **{r['trusted']}**")

        if status["rounds_played"] >= 3:
            df = pd.DataFrame(agent.round_log)
            st.write("")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["round"], y=df["win_rate"], mode="lines", name="Mirror's win rate",
                                      line=dict(color=ACCENT, width=3)))
            fig.add_hline(y=1 / 3, line_dash="dash", line_color=TIE_COLOR, annotation_text="33% random baseline")
            fig.update_layout(title="Win rate over time", xaxis_title="Round", yaxis_title="Cumulative win rate", yaxis_range=[0, 1])
            st.plotly_chart(plotly_layout(fig), width="stretch")

            with st.expander(f"Round-by-round history ({status['rounds_played']} rounds)"):
                hist_df = pd.DataFrame(agent.history).iloc[::-1].reset_index(drop=True)
                hist_df.index += 1
                st.dataframe(hist_df, width="stretch")

    with col_mind:
        st.subheader("What I've learned about you")
        status = agent.status()
        if status["rounds_played"] == 0:
            st.info("Play a few rounds. I need a handful of throws before any pattern shows up.")
        else:
            st.markdown(f'<div class="trust-card">\U0001F4AC {agent.narration()}</div>', unsafe_allow_html=True)

            st.markdown("**Confidence I'm reading you correctly**")
            st.progress(min(status["confidence"], 1.0))
            st.caption(f'{status["confidence"]*100:.0f}% recent accuracy from my trusted model')

            st.markdown(f"**Currently trusting:** {status['trusted_predictor']}")

            st.markdown("**All predictors, live accuracy on you (last 10 rounds):**")
            trust_df = pd.DataFrame(status["trust_scores"], columns=["Predictor", "Accuracy"]).sort_values("Accuracy", ascending=True)
            fig = px.bar(trust_df, x="Accuracy", y="Predictor", orientation="h", range_x=[0, 1],
                         color="Accuracy", color_continuous_scale=["#ef4444", "#94a3b8", "#22c55e"])
            fig.add_vline(x=1 / 3, line_dash="dash", line_color="white", opacity=0.4)
            fig.update_layout(coloraxis_showscale=False, title=None)
            st.plotly_chart(plotly_layout(fig, height=260), width="stretch")

# =====================================================================
# TAB 2 — Live Mind Model
# =====================================================================
with tab_mind:
    st.subheader("\U0001F9E0 What the Q-table actually learned")
    st.caption(
        "State = the recent accuracy bucket (na/low/med/high) of every predictor. "
        "Action = which predictor to trust. Each row below is a state the agent has actually "
        "visited this session; the color is the learned Q-value for trusting that predictor in that state."
    )
    labels, names, matrix = agent.q_table_frame()
    if not labels:
        st.info("No states visited yet — play a few rounds in the Play tab first.")
    else:
        heat = go.Figure(data=go.Heatmap(
            z=matrix, x=names, y=labels, colorscale="Purples", colorbar=dict(title="Q-value"),
        ))
        heat.update_layout(title=f"Q-table ({len(labels)} visited states)", xaxis_title="Predictor (action)", yaxis_title="State (predictor accuracy buckets)")
        st.plotly_chart(plotly_layout(heat, height=max(280, 28 * len(labels))), width="stretch")

    st.write("---")
    st.subheader("\U0001F4C8 Predictor trust over time")
    if len(agent.trust_log) < 2:
        st.info("Play a few more rounds to see trust curves diverge.")
    else:
        trust_hist = pd.DataFrame(agent.trust_log)
        trust_hist.insert(0, "round", range(1, len(trust_hist) + 1))
        fig = go.Figure()
        for col in trust_hist.columns[1:]:
            fig.add_trace(go.Scatter(x=trust_hist["round"], y=trust_hist[col], mode="lines", name=col))
        fig.add_hline(y=1 / 3, line_dash="dash", line_color=TIE_COLOR)
        fig.update_layout(title="Each predictor's windowed accuracy on you, round by round", xaxis_title="Round", yaxis_title="Accuracy (last 10 rounds)", yaxis_range=[0, 1])
        st.plotly_chart(plotly_layout(fig), width="stretch")

# =====================================================================
# TAB 3 — Simulation Lab
# =====================================================================
with tab_sim:
    st.subheader("\U0001F916 Run the agent against a simulated opponent, live")
    st.caption(
        "Instead of waiting on real report generation, run any agent kind against any simulated "
        "human archetype right here and watch the outcome — this is the same comparison "
        "`evaluate.py` runs offline for the report, exposed as an interactive tool."
    )

    c1, c2, c3 = st.columns(3)
    archetype_name = c1.selectbox("Simulated opponent archetype", list(ARCHETYPES.keys()))
    agent_label = c2.selectbox("Agent / baseline to test", list(SIM_AGENT_KINDS.keys()))
    n_rounds = c3.slider("Rounds per trial", 20, 300, 100, step=10)
    n_trials = st.slider("Trials to average (reduces noise, like evaluate.py does)", 1, 20, 5)

    if st.button("▶ Run simulation", type="primary"):
        agent_kind = SIM_AGENT_KINDS[agent_label]
        archetype_fn = ARCHETYPES[archetype_name]
        curves = []
        last_history = None
        for _ in range(n_trials):
            curve, hist = run_game(agent_kind, archetype_fn, n_rounds)
            curves.append(curve)
            last_history = hist
        avg_curve = pd.DataFrame(curves).mean(axis=0).tolist()
        final_wins = int(round(avg_curve[-1] * n_rounds))
        alternative = "two-sided" if archetype_name == "random" else "greater"
        p_value = exact_binomial_pvalue(final_wins, n_rounds, alternative=alternative)
        st.session_state.sim_result = {
            "archetype": archetype_name, "agent_label": agent_label, "curves": curves,
            "avg_curve": avg_curve, "history": last_history, "n_rounds": n_rounds,
            "n_trials": n_trials, "p_value": p_value,
        }

    result = st.session_state.sim_result
    if result:
        st.write("---")
        k1, k2, k3 = st.columns(3)
        k1.metric("Final win rate (avg)", f"{result['avg_curve'][-1]*100:.1f}%", delta=f"{(result['avg_curve'][-1] - 1/3)*100:+.1f} pts vs 33%")
        k2.metric("Trials averaged", result["n_trials"])
        sig = "significant (p<0.05)" if result["p_value"] < 0.05 else "not significant"
        k3.metric("Exact binomial p-value", f"{result['p_value']:.4f}", delta=sig, delta_color="off")

        fig = go.Figure()
        for i, curve in enumerate(result["curves"]):
            fig.add_trace(go.Scatter(y=curve, mode="lines", line=dict(color=ACCENT, width=1), opacity=0.25, showlegend=False))
        fig.add_trace(go.Scatter(y=result["avg_curve"], mode="lines", name="mean win rate", line=dict(color=ACCENT, width=3)))
        fig.add_hline(y=1 / 3, line_dash="dash", line_color=TIE_COLOR, annotation_text="33% baseline")
        fig.update_layout(title=f"{result['agent_label']} vs {result['archetype']} — {result['n_trials']} trials × {result['n_rounds']} rounds",
                           xaxis_title="Round", yaxis_title="Cumulative win rate", yaxis_range=[0, 1])
        st.plotly_chart(plotly_layout(fig), width="stretch")

        st.write("---")
        st.markdown("**This session, projected onto the behavioral PCA map**")
        st.caption("The session you just ran is added as a ★ marker over a freshly-sampled cloud of the 6 archetypes' behavioral fingerprints.")
        feature_rows, cloud_labels = [], []
        for arche_name, arche_fn in ARCHETYPES.items():
            for _ in range(12):
                _, h = run_game("mirror", arche_fn, 60)
                feature_rows.append(behavioral_features(h))
                cloud_labels.append(arche_name)
        session_feat = behavioral_features(result["history"])
        feature_rows.append(session_feat)
        cloud_labels.append(f"THIS SESSION ({result['agent_label']})")

        coords, explained = fit_pca(feature_rows)
        plot_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
        plot_df["archetype"] = cloud_labels
        plot_df["is_session"] = plot_df["archetype"].str.startswith("THIS SESSION")

        fig2 = px.scatter(
            plot_df[~plot_df["is_session"]], x="PC1", y="PC2", color="archetype", opacity=0.6,
            labels={"PC1": f"PC1 ({explained[0]*100:.0f}% var)", "PC2": f"PC2 ({explained[1]*100:.0f}% var)"},
        )
        session_row = plot_df[plot_df["is_session"]]
        fig2.add_trace(go.Scatter(x=session_row["PC1"], y=session_row["PC2"], mode="markers",
                                   marker=dict(symbol="star", size=22, color="white", line=dict(color=ACCENT, width=2)),
                                   name="this session"))
        fig2.update_layout(title="Behavioral feature space (this session vs archetype clouds)")
        st.plotly_chart(plotly_layout(fig2, height=440), width="stretch")
    else:
        st.info("Pick an archetype and agent, then run a simulation to see results here.")

# =====================================================================
# TAB 4 — About
# =====================================================================
with tab_about:
    st.subheader("Problem statement")
    st.markdown(
        """
Humans asked to play "randomly" systematically fail to — they repeat winning moves, switch
after losing, over-favor certain moves, and over-correct into detectable cyclic patterns when
trying to look unpredictable. **The Mirror** is a reinforcement-learning agent that discovers
*which* of these tells a specific opponent is exhibiting, purely through play (no labeled
dataset, no pretraining), and exploits it to win above the 33% random-chance baseline.
        """
    )

    st.subheader("Reinforcement learning setup")
    st.markdown(
        """
| | |
|---|---|
| **Environment** | Live RPS match vs. a human (Play tab) or a simulated archetype (Simulation Lab / `evaluate.py`) |
| **Agent** | `MirrorAgent` — tabular Q-learning meta-agent (`agent.py`) |
| **State** | Recent (last 10 rounds) accuracy of each of 5 behavioral predictors, discretized into na/low/med/high |
| **Action** | Which predictor to trust this round |
| **Reward** | +1 win, −1 loss, −0.2 tie |
        """
    )

    st.subheader("The predictor ensemble")
    pred_table = pd.DataFrame(
        [
            ("Frequency", "Overall move bias (e.g. over-playing Rock)"),
            ("Markov (last-move)", "P(next move | last move)"),
            ("Markov (last-2)", "P(next move | last two moves) — longer, more deliberate habits"),
            ("Outcome Reaction", "Learns YOUR stay/shift tendency after a win vs. after a loss separately — catches classic win-stay/lose-shift, its opposite (lose-stay), or either-direction repeaters"),
            ("Anti-Repeat", "Over-corrected 'trying to look random' cycling"),
        ],
        columns=["Predictor", "Behavioral tell it detects"],
    )
    st.dataframe(pred_table, hide_index=True, width="stretch")
    st.caption("The Q-learning layer is never told which predictor is best — it discovers that live, per-opponent, from reward alone.")

    st.subheader("Dimensionality reduction (PCA)")
    st.markdown(
        "Each session is reduced to a 6-D behavioral feature vector (3 move frequencies + "
        "stay-after-win rate + shift-after-loss rate + repeat rate), standardized, and projected "
        "to 2D with PCA — used both offline (`evaluate.py` → `results/pca_player_types.png`) and "
        "live in the Simulation Lab tab, to show that different play styles occupy separable "
        "regions of behavioral space rather than applying PCA for its own sake."
    )
    results_dir = Path("results")
    img_cols = st.columns(2)
    shown = False
    for i, name in enumerate(["pca_player_types.png", "winrate_curves.png"]):
        f = results_dir / name
        if f.exists():
            img_cols[i % 2].image(str(f), caption=name, width="stretch")
            shown = True
    if not shown:
        st.caption("Run `python evaluate.py` to generate the offline report figures shown here.")

    st.subheader("Evaluation & algorithm comparison")
    st.markdown(
        """
`evaluate.py` compares, over 200-round games × 15 trials, against 6 simulated archetypes
(random, rock-biased, win-stay/lose-shift, cyclic, anti-repeat):

- **MirrorAgent** (full Q-learning ensemble)
- **UCB1BanditAgent** (contextual UCB1 over the same predictor actions, no bootstrapping)
- **RandomAgent** (floor baseline — should stay at ~33% everywhere)
- **Each single predictor used alone**, with no meta-learning

...plus an α/ε hyperparameter sensitivity grid and exact binomial significance tests against
the 33% baseline. The Simulation Lab tab exposes this exact same comparison live, so you don't
have to leave the app to see it.
        """
    )
    comp_path = results_dir / "comparison_table.csv"
    if comp_path.exists():
        st.dataframe(pd.read_csv(comp_path), width="stretch", hide_index=True)

    st.subheader("Rubric mapping")
    st.markdown(
        """
| Rubric item | Where it's satisfied |
|---|---|
| Problem definition, dataset/environment, EDA | Problem statement above; environment = live/simulated RPS; EDA = archetype behavioral-feature distributions (Simulation Lab, `results/behavioral_features.csv`) |
| Algorithm + dimensionality reduction, evaluation | Q-learning (`agent.py`), PCA (`analysis.py`), significance tests + hyperparameter sweep (`evaluate.py`) |
| Algorithm comparison | Mirror vs UCB1 vs Random vs single predictors (Simulation Lab + `evaluate.py`) |
| Real-time GUI, interactive | This app — live play, live Q-table, live simulation, no pre-computed results |
        """
    )

    st.caption("Built for the Real-Time ML project — Reinforcement Learning approach.")
