"""Overfit or Not? — a game for discovering data science concepts by playing.

Run from partB-game/:
    streamlit run app/main.py
"""
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `levels` importable
sys.path.insert(0, str(Path(__file__).resolve().parent))      # make `logger` importable

import streamlit as st

from logger import make_logger
from levels import LEVELS

SEED = 42

st.set_page_config(page_title="Overfit or Not?", page_icon="🎯", layout="centered")

@st.cache_resource(show_spinner="Training the models for every level…")
def build_levels():
    """Levels hold only data and cached computations, so all players can share them.
    Each player's progress lives in their own session state."""
    return [cls(seed=SEED) for cls in LEVELS]


ss = st.session_state
ss.levels = build_levels()
if "session_id" not in ss:
    ss.session_id = uuid.uuid4().hex[:8]
    ss.states = {lvl.key: lvl.new_state() for lvl in ss.levels}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.title("🎯 Overfit or Not?")
    player = st.text_input("Your name or nickname (for the playtest log)", key="player")
    st.markdown("""**How to play**
1. Each level gives you real data, a few controls and a goal.
2. You see how your model does on the data it trained on.
3. **Submit** to reveal how it does on hidden data it never saw.
4. Earn stars to unlock the next level, then write one sentence on what you learned.""")
    st.caption(f"Session {ss.session_id} · seed {SEED}")

if not player:
    st.title("Overfit or Not?")
    st.write("You're a junior data scientist. Your models will be judged on data they "
             "have **never seen**. Enter a name in the sidebar to start.")
    st.stop()

log = make_logger(ss.session_id, player)

# ------------------------------------------------------------------ level picker
unlocked = [ss.levels[0]]
for prev, nxt in zip(ss.levels, ss.levels[1:]):
    if ss.states[prev.key].passed:
        unlocked.append(nxt)
    else:
        break

names = [lvl.title for lvl in unlocked]
locked = [f"🔒 {t}" for lvl in ss.levels[len(unlocked):] for t in [lvl.title]]
choice = st.radio("Level", names + locked, horizontal=False, label_visibility="collapsed")
if choice not in names:
    st.info("Pass the earlier levels to unlock this one.")
    st.stop()

level = unlocked[names.index(choice)]
state = ss.states[level.key]

st.header(level.title)
st.markdown(f"**Goal:** {level.goal}")
with st.expander("How this level works"):
    st.write(level.how_to_play)
    st.write(f"**You'll learn:** {level.objective}")

level.render(state, log)
