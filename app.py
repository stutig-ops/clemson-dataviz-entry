import streamlit as st
import pandas as pd
import plotly.express as px

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="ML Algorithm Selection Quadrant for Construction - Graduate Student Data Visualization Competition 2026 - Stuti Garg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. HEADER & ABSTRACT ---
st.title("ML Algorithm Selection Quadrant for Construction Industry")
st.markdown("""
**Visualization Narrative:** In a systematic literature review investigating 30 studies discussing applications of ML algorithms for construction industry 113 algorithmic implementations were grouped in 11 categories. 
A trade-off was identified and each algorithmic family was mapped to the quadrant, with the horizontal axis representing the average Complexity Fit (C) per family and the vertical axis representing the average Data Fit (D). 
X Axis: **Complexity Fit (C)** - measures the ability to capture complex, non‑linear, and high‑dimensional relationships
Y-Axis: **Data Fit (D)** - measures robustness to real-world construction data challenges - missing values, class imbalance, sample‑size variation
* **Quadrant 1:** Advanced and Sophisticated: (high C, high D) – Best of Both: Random Forest, Boosting/Gradient, and Ensemble families form a prominent cluster combining strong non‑linear modeling with comparatively better robustness to missing values, imbalance, and varying sample sizes. These families also have the largest bubbles, reflecting their high empirical maturity/adoption in the construction ML literature.
* **Quadrant 2:** Simple and Robust: (low C, high D) Regression, Naïve-Bayesian, and Decision trees represent the foundational methods that remain critical as baseline comparisons. These methods are better suited to linear relationships within variables/features and rely on clean, balanced data; they are unable to capture the inherent nonlinearity of construction data.
* **Quadrant 3:** Limited Applicability: (low C, low D) KNN models show weaker data fit despite modest complexity, suggesting applicability to a limited application.
* **Quadrant 4:** Complex and Fragile: (high C, low D) ANN scores exceptionally high in their ability to manage complex dataset interactions, with not much emphasis on handling missing data or rare event predictions. SVM, on the other hand, with kernel tricks, can handle multi-dimensional datasets that require synthetic oversampling to mitigate imbalanced classes.
""")

# --- 3. DATA LOADING & PRE-PROCESSING ---
@st.cache_data
def load_and_jitter_data():
    # Load the raw data (113 rows)
    df = pd.read_csv('algo_table_with_scores.csv')
    
    # Map raw category names to clean names
    name_map = {
        'Artificial Neural Networks (ANN)': 'ANN',
        'Bayesian networks': 'Bayesian Networks',
        'Boosting/Gradient ': 'Boosting/Gradient',
        'Boosting/Gradient': 'Boosting/Gradient',
        'Decision Tree': 'Decision Tree',
        'Ensemble': 'Ensemble',
        'Extremely Randomized Trees': 'Extremely Randomized Trees',
        'Naïve-Bayesian Classifier': 'Naïve-Bayesian',
        'Naïve-Bayesian Classifier ': 'Naïve-Bayesian',
        'Random Forest': 'Random Forest',
        'Regression': 'Regression',
        'Support Vector Machine (SVM)': 'SVM',
        'k-Nearest Neighbour (KNN)': 'KNN'
    }
    df['category_clean'] = df['category'].map(name_map).fillna(df['category'])

    # ADD JITTER (Deterministic)
    # This separates overlapping points so we can see the "Clusters"
    np.random.seed(42) # Reproducible seed
    jitter_strength = 0.35
    df['X_Jittered'] = df['X_complexity'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    df['Y_Jittered'] = df['Y_sophistication'] + np.random.uniform(-jitter_strength, jitter_strength, len(df))
    
    return df

df = load_and_jitter_data()

# --- 4. SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Configuration")

# A. Task Context Selector
task_context = st.sidebar.radio(
    "Select Task Context:",
    ("General Overview", "Safety Management", "Schedule Optimization", "Cost Prediction"),
    help="Resize bubbles based on their suitability score for the selected task."
)

st.sidebar.divider()

# B. Method Highlighter (Optional Filter)
st.sidebar.subheader("🔍 Filter by Family")
algo_options = ["All Families"] + sorted(df['category_clean'].unique())
selected_family = st.sidebar.selectbox("Highlight Family:", algo_options, index=0)

# Details Panel
if selected_family != "All Families":
    # Get stats for the family
    subset = df[df['category_clean'] == selected_family]
    avg_safety = subset['safety_suitability'].mean()
    avg_sched = subset['schedule_suitability'].mean()
    avg_cost = subset['cost_suitability'].mean()
    count = len(subset)

    st.sidebar.subheader(f"📊 {selected_family}")
    st.sidebar.caption(f"**{count}** Implementations found.")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Avg Safety", f"{avg_safety:.2f}")
    with col2:
        st.metric("Avg Cost", f"{avg_cost:.2f}")
    st.sidebar.metric("Avg Schedule", f"{avg_sched:.2f}")

# --- 5. VISUALIZATION LOGIC ---

# Define Size Variable based on Context
if task_context == "General Overview":
    # In General mode, use Maturity or a fixed size
    df['Size_Var'] = df['Maturity_M'] * 2 # Scale up for visibility
    df['Size_Label'] = "Maturity: " + df['Maturity_M'].astype(str)
    size_title = "Maturity (M)"
elif task_context == "Safety Management":
    df['Size_Var'] = df['safety_suitability'] * 20 # Scale factor for bubbles
    df['Size_Label'] = df['safety_suitability'].round(2).astype(str)
    size_title = "Safety Score"
elif task_context == "Schedule Optimization":
    df['Size_Var'] = df['schedule_suitability'] * 20
    df['Size_Label'] = df['schedule_suitability'].round(2).astype(str)
    size_title = "Schedule Score"
elif task_context == "Cost Prediction":
    df['Size_Var'] = df['cost_suitability'] * 20
    df['Size_Label'] = df['cost_suitability'].round(2).astype(str)
    size_title = "Cost Score"

# Professional Muted Pastel Palette
pastel_map = {
    'Artifical Neural Network (ANN)': '#D68C9F', # Deep Dusty Rose (Darker/Redder)
    'Bayesian Networks': '#A6C6CC',      # Powder Teal
    'Boosting/Gradient': '#A3C1A3',      # Sage Green
    'Decision Tree': '#BFB5C2',          # Lilac Grey
    'Ensemble': '#E6C8C8',               # Dusty Rose
    'Extremely Randomized Trees': '#D1D1AA', # Khaki Pastel
    'k-Nearest Neighbor (KNN)': '#9FA8DA',                    # Muted Periwinkle (Now distinctly BLUE, not grey)
    'Naïve-Bayesian': '#C4AFAF',         # Mauve Taupe
    'Random Forest': '#DDB8AC',          # Peach Grey
    'Regression': '#ABC6D4',             # Slate Blue Pastel
    'Support Vector Machine (SVM)': '#78909C'                     # Blue Grey (Darker and distinct from the background)
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
        'category_clean': True, 'Size_Label': True, 'objective': True
    },
    size_max=40, # Maximum bubble size
    template="plotly_white",
    labels={
        "X_Jittered": "Algorithm Complexity (X)",
        "Y_Jittered": "Algorithm Sophistication (Y)",
        "category_clean": "Algorithm Family"
    }
)

# --- APPLY FORMATTING & QUADRANTS ---

# 1. Spotlight Logic (Dimming unselected families)
if selected_family != "All Families":
    for trace in fig.data:
        if trace.name == selected_family:
            trace.marker.opacity = 0.8
            trace.marker.line.width = 1
            trace.marker.line.color = 'black'
        else:
            trace.marker.opacity = 0.1 # Dim others
            trace.marker.line.width = 0
else:
    # Default opacity for clusters
    fig.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))

# 2. Add Quadrant Lines (Center at 5.5, 5.5 as per your Python code)
center_val = 5.5
fig.add_vline(x=center_val, line_width=2, line_dash="dash", line_color="grey")
fig.add_hline(y=center_val, line_width=2, line_dash="dash", line_color="grey")

# 3. Add Quadrant Labels (Styled like your Matplotlib code)
# Quadrant 1 (Top Left) -> Simple & Basic? Wait, X is Complexity.
# Low X (Simple), High Y (Sophisticated) -> Top Left
fig.add_annotation(x=2.5, y=8.5, text="<b>Simple &<br>Sophisticated</b>", showarrow=False, 
                   bgcolor="#e8f4f8", bordercolor="grey", borderwidth=1, opacity=0.8)

# Quadrant 2 (Top Right) -> High X (Complex), High Y (Sophisticated)
fig.add_annotation(x=7.5, y=8.5, text="<b>Advanced &<br>Sophisticated</b>", showarrow=False, 
                   bgcolor="#e8f8e8", bordercolor="grey", borderwidth=1, opacity=0.8)

# Quadrant 3 (Bottom Left) -> Low X (Simple), Low Y (Basic)
fig.add_annotation(x=2.5, y=1.5, text="<b>Simple &<br>Basic</b>", showarrow=False, 
                   bgcolor="#f8e8e8", bordercolor="grey", borderwidth=1, opacity=0.8)

# Quadrant 4 (Bottom Right) -> High X (Complex), Low Y (Basic)
fig.add_annotation(x=7.5, y=1.5, text="<b>Complex but<br>Established</b>", showarrow=False, 
                   bgcolor="#ffffe0", bordercolor="grey", borderwidth=1, opacity=0.8)

# 4. Final Layout Config
fig.update_layout(
    height=750,
    margin=dict(l=40, r=40, t=60, b=40),
    xaxis=dict(range=[0, 10.5], showgrid=True, dtick=1, title_font=dict(size=16, family="Arial Black")),
    yaxis=dict(range=[0, 10.5], showgrid=True, dtick=1, title_font=dict(size=16, family="Arial Black")),
    legend=dict(title="Algorithm Family", orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# --- 6. METHODOLOGY FOOTER ---
st.divider()
st.caption("""
**Methodology:** Data derived from a meta-analysis of 30 empirical studies. Scores (C, D, I, M) were calculated based on 11 algorithmic indicators.
A systematic literature review following PRISMA guidelines analyzed 30 articles encompassing 113 algorithms used for construction applications. 
The development of the framework was a result of a four-stage process:
(1) synthesis and extraction of algorithmic characteristics from the findings of the SLR
(2) systematic coding of algorithms implementations based on the theoretical and empirical knowledge of performance patterns of the algorithms
(3) development of a multidimensional scoring framework
(4) quadrant-based visualization of the algorithms between model complexity, dataset characteristics and frequency of adoption

**Visual Encoding:**
* **X-Axis:** Complexity (1-10) - Ability to handle non-linearity.
* **Y-Axis:** Sophistication (1-10) - Ability to handle real-world data issues (missing values, imbalance).
* **Bubble Size:** {size_title}
* **Clusters:** 113 distinct algorithmic implementations from the literature, jittered for visibility.
""")

For full reproducibility, view the [Source Code & Analysis Pipeline](https://github.com/stutig-ops/clemson-dataviz-entry).

""")












