# ML Algorithm Selection Quadrant for Construction - Stuti Garg
**Graduate Student Data Visualization Competition 2026**

### 🔗 Live Dashboard
[Click here to launch the interactive App](https://clemson-dataviz-entry-mlalgoclusters.streamlit.app/)

### Overview
Machine Learning (ML) in construction often presents a “paradox of choice,” making it difficult for practitioners to determine which algorithm best fits their data and task. 
A systematic literature review (SLR) of 30 foundational studies revealed that 39% of algorithms in construction research are selected based on precedent, while 16% lack any stated rationale (Garg et al., 2025). 
Grounded in 113 algorithmic implementations, the ML Algorithm Selection Quadrant provides a prescriptive, data‑driven framework to guide construction researchers and practitioners. 
It maps algorithms across two empirically derived dimensions—Complexity Fit (C) and Data Fit (D)—that reflect both their modeling capabilities and their robustness to challenging construction datasets. 
The quadrant serves primarily three goals:
* **“Bridge the Gap”** - Democratizing ML selection: Translating abstract algorithmic properties into a concrete, actionable framework beyond trial-and-error methods
* **Expose hidden “trade-offs”:** Accuracy is not the only parameter for selecting a technique, exhibiting how high complexity often comes at the cost of data fragility.
* **Decision-making support:** Evidence-based guide to help researchers and practitioners select the right algorithm aligned with a specific dataset and task, rather than popularity.


## Methodology (Reproducibility)
This visualization was built using an open-source Python stack to ensure full transparency and reproducibility.

* **Data Processing:** **Pandas & NumPy** for data cleaning, score aggregation, and jittering logic.
* **Visualization:** **Plotly Express** for the interactive quadrant chart.
* **Deployment:** **Streamlit** for the web interface.

## Reproducibility
* **Dashboard Code:** `app.py`
* **Analysis Pipeline:** `analysis_pipeline.ipynb` (Contains the full data cleaning and score calculation logic).

## How to Run Locally
1.  Clone this repository.
2.  Install dependencies: `pip install -r requirements.txt`
3.  Run the app: `streamlit run app.py`

## 📂 Files
* `app.py`: The main source code for the visualization dashboard.
* `algo_table_with_scores.csv`: The processed dataset used for the analysis.
