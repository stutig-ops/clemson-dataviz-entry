# ML Algorithm Selection Quadrant for Construction Industry - Stuti Garg
**Graduate Student Data Visualization Competition 2026**

### 🔗 Live Dashboard
[Click here to launch the interactive App](https://clemson-viz-entry-ah2k4qgwmdeww8d7ao6z75.streamlit.app/)

### Overview
This project challenges the "accuracy is everything" mindset in construction AI. By synthesizing data from 30+ empirical studies, and mapping algorithms on two axes: Complexity Fit vs. Data Fit - the visual demonstrates that **Ensemble methods** (like Random Forest) are the industry "Gold Standard" because they balance complexity with the robustness needed for imperfect construction data. It synthesizes data from 30+ studies to 

## Methodology (Reproducibility)
This visualization was built using an open-source Python stack to ensure full transparency and reproducibility.
* **Data Processing:** `Pandas` for aggregating complexity/data-fit scores.
* **Visualization:** `Plotly Express` for the interactive quadrant chart.
* **Deployment:** `Streamlit` for the web interface.

### Reproducibility
* **Dashboard Code:** `app.py`
* **Analysis Pipeline:** `analysis_pipeline.ipynb` (Contains the full data cleaning and score calculation logic).

### How to Run Locally
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

## 📂 Files
* `app.py`: The main source code for the visualization dashboard.
* `algo_table_with_scores.csv`: The processed dataset used for the analysis.
