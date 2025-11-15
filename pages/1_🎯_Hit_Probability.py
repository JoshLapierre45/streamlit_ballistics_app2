import streamlit as st
from ballistics_explorer.hit_probability_tab import render as hitprob_tab

st.set_page_config(page_title="Hit Probability", layout="wide")
hitprob_tab()
