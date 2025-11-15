# pages/1_📄_Bio.py
import streamlit as st

st.set_page_config(page_title="Bio", layout="wide")
st.title("📄 Professional Bio")

col1, col2 = st.columns([1,2])
with col1:
    st.image("Data/Headshot.jpg", use_container_width=True)
with col2:
    st.subheader("Hi, I’m Josh Lapierre")
    st.write("""
I’m a data practitioner focused on applied analytics and interactive visualization. 
I enjoy turning messy, real-world data into useful tools, especially when accuracy, 
clarity, and speed matter (like ballistics!). I work primarily with **Python**, **Pandas**, 
**Altair/Plotly**, and **Streamlit**, and I’m comfortable with scikit-learn for modeling.
    """)
    st.write("""
**Highlights**
- Coursework: Data Visualization, Statistics, ML fundamentals
- Tools: Python, Pandas, Altair, Plotly, Streamlit, scikit-learn
- Interests: Ballistics modeling, calibration/uncertainty, dashboards
- Strengths: Clear storytelling, reproducible analysis
    """)

