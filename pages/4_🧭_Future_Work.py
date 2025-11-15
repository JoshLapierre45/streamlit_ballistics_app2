# pages/4_🧭_Future_Work.py
import streamlit as st

st.set_page_config(page_title="Future Work", layout="wide")
st.title("🧭 Future Work")

st.subheader("Next Steps")
st.markdown("""
1. **Forecasting**: add a range-based hit-probability curve that extrapolates with confidence bands.  
2. **A/B Layouts**: test alternative dashboard layouts (cards on top vs left rail).  
3. **Data Enrichment**: add measured crosswind/headwind components at the range.  
4. **Accessibility Audit**: run color-contrast checks; add text alternatives in all charts.  
5. **Modeling**: switch to calibrated boosted trees for better nonlinearity and calibration.
""")

st.subheader("Reflection")
st.markdown("""
- Moved from a single-page prototype to a clear **multipage** app.  
- Replaced ad-hoc plots with a consistent **EDA Gallery** and **Dashboard** that share filters.  
- Added **probability modeling** with log-loss evaluation and explained calibration choices.  
- Documented **ethics & accessibility** and provided download templates and data dictionary.
""")

st.subheader("Ethics Note (People Data)")
st.info("""
This dataset includes observations related to people (shooters), so results must be interpreted carefully. 
Because the data come from a limited context and may include measurement noise or selection bias, 
they do not necessarily represent all shooters or conditions. The visualizations show patterns and 
uncertainty but should not be used to make individual judgments or broad generalizations.
""")
