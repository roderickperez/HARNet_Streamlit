import streamlit as st
import pandas as pd

st.set_page_config(
     page_title = "HARNet | Research Project",
     page_icon=":chart_with_upwards_trend:",
     layout="wide",
     initial_sidebar_state="expanded"
 )

# ---- Header ----

st.write("HARNet Project")

with st.container():
    st.subheader("University of Vienna")
    st.title("HARNet | Research Project")
    st.write("Reference: [HARNet](https://arxiv.org/abs/1903.04909)")
    