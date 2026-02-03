import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ML Algorithm Selection Quadrant for Construction - Graduate Student Data Visualization Competition 2026 - Stuti Garg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HEADER & ABSTRACT ---
st.title("ML Algorithm Selection Quadrant for Construction")
st.markdown("""
**Visualization Narrative:** In a systematic literature review investigating 30 studies discussing applications of ML algorithms for construction industry 113 algorithmic implementations were grouped in 10 categories. 
A trade-off was identified and each algorithmic family was mapped to the quadrant, with the horizontal axis representing the average Complexity Fit (C) per family and the vertical axis representing the average Data Fit (D). 
X Axis: **Complexity Fit (C)** - measures the ability to capture complex, non‑linear, and high‑dimensional relationships
Y-Axis: **Data Fit (D)** - measures robustness to real-world construction data challenges - missing values, class imbalance, sample‑size variation
* **Quadrant 1:** Advanced and Sophisticated: (high C, high D) – Best of Both: Strong non‑linear modeling with comparatively better robustness to missing values, imbalance, and varying sample sizes. These families also have the largest bubbles, reflecting their high empirical maturity/adoption in the construction ML literature.
* **Quadrant 2:** Simple and Robust: (low C, high D) These methods are better suited to linear relationships within variables/features and rely on clean, balanced data; they are unable to capture the inherent nonlinearity of construction data.
* **Quadrant 3:** Limited Applicability: (low C, low D) Weaker data fit despite modest complexity, suggesting applicability to specific objective.
* **Quadrant 4:** Complex and Fragile: (high C, low D) Exceptionally high in their ability to manage complex dataset interactions, with not much emphasis on handling missing data or rare event predictions. SVM with kernel tricks, can handle multi-dimensional datasets that require synthetic oversampling to mitigate imbalanced classes.
""")

# --- 3. DATA LOADING & PRE-PROCESSING ---
@st.cache_data
def load_and_process_data():
    # Load the raw data (113 rows)
    df = pd.read_csv('algo_table_with_scores.csv')
    
    # 1. Clean Category Names
    name_map = {
        'Artificial Neural Networks (ANN)': 'Artificial Neural Networks (ANN)',
        'Bayesian networks': 'Bayesian Networks',
        'Boosting/Gradient ': 'Boosting/Gradient',
        'Boosting/Gradient': 'Boosting/Gradient',
        'Decision Tree': 'Decision Tree',
        'Ensemble': 'Ensemble',
        'Naïve-Bayesian Classifier': 'Naïve-Bayesian',
        'Naïve-Bayesian Classifier ': 'Naïve-Bayesian',
        'Random Forest': 'Random Forest',
        'Regression': 'Regression',
        'Support Vector Machine (SVM)': 'Support Vector Machine (SVM)',
        'k-Nearest Neighbour (KNN)': 'k-Nearest Neighbour (KNN)'
    }
    df['category_clean'] = df['category'].map(name_map).fillna(df['category'])

    # 2. CALCULATE MATURITY (M) DYNAMICALLY
    family_counts = df['category_clean'].value_counts()
    max_freq = family_counts.max() # Should be 29
    
    df['Family_Count'] = df['category_clean'].map(family_counts)
    df['Calculated_Maturity'] = df['Family_Count'] / max_freq

    # 3. DYNAMIC COORDINATE CALCULATION
    category_stats = df.groupby('category_clean')[['complexity_fit_C', 'data_fit_D']].mean().reset_index()
    category_stats.rename(columns={
        'complexity_fit_C': 'Avg_Complexity_C', 
        'data_fit_D': 'Avg_DataFit_D'
    }, inplace=True)
    
    df = df.merge(category_stats, on='category_clean', how='left')

    # 4. ADD JITTER (Deterministic)
    np.random.seed(42) 
    jitter_strength = 0.03 
    
    df['X_Jittered'] = df['Avg_Complexity_C'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    df['Y_Jittered'] = df['Avg_DataFit_D'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    
    # Calculate Medians for Quadrant Boundaries
    x_median = df['Avg_Complexity_C'].median()
    y_median = df['Avg_DataFit_D'].median()
    
    return df, x_median, y_median

df, x_median, y_median = load_and_process_data()

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.header("Configuration")

# A. Task Context Selector
task_context = st.sidebar.radio(
    "Select Task Context:",
    ("General Overview", "Safety Management", "Schedule Optimization", "Cost Prediction"),
    help="Resize bubbles based on their suitability score for the selected task."
)

st.sidebar.divider()

# B. Method Highlighter (Optional Filter)
st.sidebar.subheader("Filter by Family")
algo_options = ["All Families"] + sorted(df['category_clean'].unique())
selected_family = st.sidebar.selectbox("Highlight Family:", algo_options, index=0)

# Details Panel
if selected_family != "All Families":
    subset = df[df['category_clean'] == selected_family]
    count = subset['Family_Count'].iloc[0]
    maturity = subset['Calculated_Maturity'].iloc[0]
    
    # Extract Scores
    fam_c = subset['Avg_Complexity_C'].iloc[0]
    fam_d = subset['Avg_DataFit_D'].iloc[0]
    avg_safety = subset['safety_suitability'].mean()
    avg_sched = subset['schedule_suitability'].mean()
    avg_cost = subset['cost_suitability'].mean()

    st.sidebar.subheader(f"{selected_family}")
    st.sidebar.caption(f"**{count}** Implementations (M={maturity:.2f})")
    
    # 1. Core Metrics (C & D)
    st.sidebar.markdown("##### Core Metrics")
    col_cd1, col_cd2 = st.sidebar.columns(2)
    with col_cd1:
        st.metric("Complexity (C)", f"{fam_c:.2f}", help="Avg. Complexity Fit")
    with col_cd2:
        st.metric("Data Fit (D)", f"{fam_d:.2f}", help="Avg. Data Fit")
    
    st.sidebar.divider()
    
    # 2. Task Suitability Scores
    st.sidebar.markdown("##### Task Suitability")
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Avg Safety", f"{avg_safety:.2f}")
    with col2:
        st.metric("Avg Cost", f"{avg_cost:.2f}")
    st.sidebar.metric("Avg Schedule", f"{avg_sched:.2f}")

# --- 5. VISUALIZATION LOGIC ---

# Define Size Variable
if task_context == "General Overview":
    df['Size_Var'] = df['Calculated_Maturity'] * 30 
    df['Size_Label'] = "Maturity: " + df['Calculated_Maturity'].round(2).astype(str)
    size_title = "Maturity (Frequency)"
elif task_context == "Safety Management":
    df['Size_Var'] = df['safety_suitability'] * 30 
    df['Size_Label'] = df['safety_suitability'].round(2).astype(str)
    size_title = "Safety Score"
elif task_context == "Schedule Optimization":
    df['Size_Var'] = df['schedule_suitability'] * 30
    df['Size_Label'] = df['schedule_suitability'].round(2).astype(str)
    size_title = "Schedule Score"
elif task_context == "Cost Prediction":
    df['Size_Var'] = df['cost_suitability'] * 30
    df['Size_Label'] = df['cost_suitability'].round(2).astype(str)
    size_title = "Cost Score"

# Professional Muted Pastel Palette
pastel_map = {
    'Artificial Neural Networks (ANN)': '#D68C9F',  # Deep Dusty Rose
    'Bayesian Networks': '#A6C6CC',                 # Powder Teal
    'Boosting/Gradient': '#A3C1A3',                 # Sage Green
    'Decision Tree': '#BFB5C2',                     # Lilac Grey
    'Ensemble': '#E6C8C8',                          # Dusty Rose
    'k-Nearest Neighbour (KNN)': '#9FA8DA',         # Muted Periwinkle
    'Naïve-Bayesian': '#C4AFAF',                    # Mauve Taupe
    'Random Forest': '#DDB8AC',                     # Peach Grey
    'Regression': '#ABC6D4',                        # Slate Blue Pastel
    'Support Vector Machine (SVM)': '#78909C'       # Blue Grey
}

# --- GENERATE CLUSTER PLOT ---
fig = px.scatter(
    df, 
    x="X_Jittered", 
    y="Y_Jittered", 
    size="Size_Var", 
    color="category_clean", 
    color_discrete_map=pastel_map,
    hover_name="algorithm_name",
    hover_data={
        'X_Jittered': False, 'Y_Jittered': False, 'Size_Var': False,
        'category_clean': True, 'Size_Label': True, 'objective': True,
        'Avg_Complexity_C': ':.2f', 'Avg_DataFit_D': ':.2f'
    },
    size_max=40,
    template="plotly_white",
    labels={
        "X_Jittered": "Complexity Fit (C)",
        "Y_Jittered": "Data Fit (D)",
        "category_clean": "ML Algorithm" # Clean label for legend
    }
)

# --- APPLY FORMATTING & QUADRANTS ---

# 1. Spotlight Logic
if selected_family != "All Families":
    for trace in fig.data:
        if trace.name == selected_family:
            trace.marker.opacity = 0.8
            trace.marker.line.width = 1
            trace.marker.line.color = 'black'
        else:
            trace.marker.opacity = 0.1
            trace.marker.line.width = 0
else:
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

# 2. Add Quadrant Lines
fig.add_vline(x=x_median, line_width=2, line_dash="dash", line_color="grey")
fig.add_hline(y=y_median, line_width=2, line_dash="dash", line_color="grey")

# 3. Add Quadrant Labels (Clean, Corner Positioning)
label_pad = 0.02

# Q2: Simple & Robust (Top-Left)
fig.add_annotation(
    x=0 + label_pad, y=1 - label_pad,
    text="Quadrant 2:<br>Simple & Robust",
    showarrow=False,
    xanchor="left", yanchor="top",
    font=dict(size=14, color="black")
)

# Q1: Advanced & Sophisticated (Top-Right)
fig.add_annotation(
    x=1 - label_pad, y=1 - label_pad,
    text="Quadrant 1:<br>Advanced & Sophisticated",
    showarrow=False,
    xanchor="right", yanchor="top",
    font=dict(size=14, color="black")
)

# Q3: Limited Applicability (Bottom-Left)
fig.add_annotation(
    x=0 + label_pad, y=0 + label_pad,
    text="Quadrant 3:<br>Limited Applicability",
    showarrow=False,
    xanchor="left", yanchor="bottom",
    font=dict(size=14, color="black")
)

# Q4: Complex & Fragile (Bottom-Right)
fig.add_annotation(
    x=1 - label_pad, y=0 + label_pad,
    text="Quadrant 4:<br>Complex & Fragile",
    showarrow=False,
    xanchor="right", yanchor="bottom",
    font=dict(size=14, color="black")
)

# 4. Final Layout Config (UPDATED LEGEND)
fig.update_layout(
    height=750,
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(range=[-0.1, 1.1], showgrid=True, title_font=dict(size=16, family="Arial Black")),
    yaxis=dict(range=[-0.1, 1.1], showgrid=True, title_font=dict(size=16, family="Arial Black")),
    
    # --- LEGEND CONFIGURATION ---
    legend=dict(
        title=dict(text="<b>ML Algorithm</b><br>", font=dict(size=18)), # Added <br> for spacing
        orientation="v",       
        yanchor="top", y=1,    
        xanchor="left", x=1.02,
        font=dict(size=15),    # Increased font slightly
        itemsizing="constant", # Ensures markers are visible
        tracegroupgap=10       
    )
)

st.plotly_chart(fig, use_container_width=True)

# --- 6. METHODOLOGY FOOTER ---
st.divider()
st.caption(f"""
**Methodology:**
Developed from a PRISMA-guided Systematic Literature Review (Garg et al. 2025) of 30 studies (113 algorithm implementations). The framework quantifies algorithm suitability through four core dimensions derived from 16 coded indicators:
1.  **Complexity Fit (C):** Ability to model non-linear/complex relationships. Calculated as: $0.4 \\times NonLinearity + 0.4 \\times ComplexPatterns + 0.2 \\times HighDimensional$.
2.  **Data Fit (D):** Robustness to data quality issues. Calculated as: $0.3 \\times MissingData + 0.3 \\times Imbalance + 0.2 \\times SmallN + 0.2 \\times LargeN$.
3.  **Interpretability (I):** Transparency level: High (1.0), Medium (0.5), or Low (0.0).
4.  **Maturity (M):** Empirical adoption frequency normalized to [0, 1].

**Task-Suitability Scoring Mechanism ($S$):** enabling users to identify which algorithms empirically perform best for the problem they are solving:
* **Safety ($S_{{safety}}$):** $0.35 \\times Imbalance + 0.25 \\times Interpretability + 0.20 \\times ComplexNonLinear + 0.20 \\times MissingData$.
* **Schedule ($S_{{schedule}}$):** $0.30 \\times Imbalance + 0.25 \\times ComplexNonLinear + 0.20 \\times LargeN + 0.15 \\times TemporalSeq + 0.10 \\times NonLinearity$.
* **Cost ($S_{{cost}}$):** $0.30 \\times NonLinearity + 0.25 \\times Interpretability + 0.20 \\times Uncertainty + 0.15 \\times Prediction + 0.10 \\times LargeN$.

**Validation:** Pearson correlation ($|r|=0.353$) confirmed C and D are orthogonal dimensions, justifying the quadrant axes defined by their median values ({x_median:.2f}, {y_median:.2f}).

**Visual Encoding:**
* **X-Axis:** Average Complexity Fit (C). Median Boundary: {x_median:.2f}
* **Y-Axis:** Average Data Fit (D). Median Boundary: {y_median:.2f}
* **Bubble Size:** {size_title}
* **Clusters:** 113 distinct algorithmic implementations, jittered for visibility.

For full reproducibility, view the [Source Code & Analysis Pipeline](https://github.com/stutig-ops/clemson-dataviz-entry).
""")











