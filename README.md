- ** (IMPORTANT!!!) To see sample data for hit probability, download the hit_probability_semi_realistic.csv from the Data/ folder and upload it on the Hit Probability Page. Change "shooter" to "josh" and "rifle" to "7mmPRC" to get a better log-loss. Change different parameters to view probability statistics.

# Ballistics Toolkit

A multipage Streamlit app for exploring ballistics data and estimating hit probability.

## App Navigation
- **Home**: overview / links to other pages
- **🎯 Hit Probability**: upload shooter history, train calibrated model, predict p(hit) vs range 
- **📊 EDA Gallery**: four chart types + “how to read” + insights
- **📈 Dashboard**: KPIs, filters, and two linked visuals
- **🧭 Future Work**: roadmap + reflection + ethics note

## Dataset
- `Data/Ballistics.xlsx` (or CSV) — >100 rows  
- 'Ballistics.xlsx' came from [Hornady](https://www.hornady.com/4dof). I chose 5 different cartridges, and 2 different bullet weights for each cartidge.
- Normalization: renamed columns (Cartridge→cartridge, etc.), computed MIL/MOA from drop + distance.

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
