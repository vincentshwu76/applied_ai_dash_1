import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="COVID Evidence Explorer",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def build_feature_vector(wealth, health, education, npis, vacc_share, income_support):
    """
    Build 14-dimensional feature vector [0,1]
    x* = [wealth, health, education, npi_1...npi_9, vacc_share, income_support]
    """
    feature_vector = [wealth, health, education] + npis + [vacc_share, income_support]
    return np.array(feature_vector)

def generate_dummy_historical_data(n_episodes=1000):
    """
    Generate dummy historical data for demonstration
    In production, load from actual COVID historical database
    """
    np.random.seed(42)
    
    historical_data = []
    for i in range(n_episodes):
        episode = {
            'id': i,
            'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Italy', 'Spain', 'Canada', 'Australia']),
            'week': np.random.randint(1, 150),
            # Feature vector components
            'wealth': np.random.uniform(0, 1),
            'health': np.random.uniform(0, 1),
            'education': np.random.uniform(0, 1),
            'npi_1': np.random.uniform(0, 1),
            'npi_2': np.random.uniform(0, 1),
            'npi_3': np.random.uniform(0, 1),
            'npi_4': np.random.uniform(0, 1),
            'npi_5': np.random.uniform(0, 1),
            'npi_6': np.random.uniform(0, 1),
            'npi_7': np.random.uniform(0, 1),
            'npi_8': np.random.uniform(0, 1),
            'npi_9': np.random.uniform(0, 1),
            'vacc_share': np.random.uniform(0, 1),
            'income_support': np.random.choice([0, 0.5, 1]),
            # Historical outcomes (per 100k)
            'cases_trajectory': np.random.uniform(10, 200, 16),  # 16 weeks
            'deaths_trajectory': np.random.uniform(0.2, 5, 16),
            'compliance': np.random.uniform(0.3, 0.9)
        }
        historical_data.append(episode)
    
    return historical_data

def find_nearest_neighbours(scenario_vector, historical_data, k=30):
    """
    Find K nearest historical episodes using Euclidean distance
    """
    distances = []
    
    for episode in historical_data:
        # Build historical episode feature vector
        hist_vector = np.array([
            episode['wealth'], episode['health'], episode['education'],
            episode['npi_1'], episode['npi_2'], episode['npi_3'],
            episode['npi_4'], episode['npi_5'], episode['npi_6'],
            episode['npi_7'], episode['npi_8'], episode['npi_9'],
            episode['vacc_share'], episode['income_support']
        ])
        
        # Compute Euclidean distance
        distance = np.sqrt(np.sum((scenario_vector - hist_vector) ** 2))
        distances.append((distance, episode))
    
    # Sort by distance and select K nearest
    distances.sort(key=lambda x: x[0])
    nearest = distances[:k]
    
    return nearest

def compute_weights(nearest_neighbours, epsilon=0.01):
    """
    Compute normalized weights emphasizing closer matches
    w_i = 1 / (d_i + epsilon)
    """
    weights = []
    for distance, episode in nearest_neighbours:
        w = 1.0 / (distance + epsilon)
        weights.append(w)
    
    # Normalize
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]
    
    return weights

def generate_projections(nearest_neighbours, weights, forecast_weeks, population):
    """
    Generate weighted average projections
    """
    n_weeks = forecast_weeks
    cases_per_100k = np.zeros(n_weeks)
    deaths_per_100k = np.zeros(n_weeks)
    compliance_weighted = 0
    
    for (distance, episode), weight in zip(nearest_neighbours, weights):
        # Average trajectories (per 100k)
        cases_per_100k += weight * episode['cases_trajectory'][:n_weeks]
        deaths_per_100k += weight * episode['deaths_trajectory'][:n_weeks]
        
        # Average compliance
        compliance_weighted += weight * episode['compliance']
    
    # Compute mortality rate
    mortality = deaths_per_100k / (cases_per_100k + 0.001)  # avoid division by zero
    
    # Scale to population
    scaling_factor = population / 100000
    cases_absolute = cases_per_100k * scaling_factor
    deaths_absolute = deaths_per_100k * scaling_factor
    
    return {
        'cases_per_100k': cases_per_100k,
        'deaths_per_100k': deaths_per_100k,
        'cases_absolute': cases_absolute,
        'deaths_absolute': deaths_absolute,
        'mortality': mortality,
        'compliance': compliance_weighted
    }

def calculate_stringency(npis):
    """
    Calculate stringency index (0-100) based on OxCGRT method
    Stringency = (1/9) * sum(100 * npi_j)
    """
    sub_indices = [100 * npi for npi in npis]
    stringency = sum(sub_indices) / 9
    return stringency

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================
if 'historical_data' not in st.session_state:
    st.session_state.historical_data = generate_dummy_historical_data()

# ============================================================================
# HEADER
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# COVID Evidence Explorer")

st.write("")

_, exp_col, _ = st.columns([1,3,1])
with exp_col:
    with st.expander("**📖 Guide to This Dashboard**"):
        st.markdown("""
<div class="info-box">
<h3>How This Dashboard Works</h3>
<p><strong>This dashboard does not use a simulation model.</strong></p>
<p>Instead, it looks for <strong>real historical COVID situations</strong> that match the settings you choose.</p>

<p><strong>You define a hypothetical country's:</strong></p>
<ul>
    <li>Development level (wealth, health, education)</li>
    <li>Vaccination rate</li>
    <li>Economic support</li>
    <li>Non-pharmaceutical interventions (NPIs)</li>
    <li>Population size</li>
</ul>

<p>These inputs form a <strong>profile vector</strong>.</p>

<p><strong>The system then:</strong></p>
<ol>
    <li>Compares your profile to thousands of real (country, week) episodes from COVID history</li>
    <li>Finds the <strong>K nearest neighbours</strong> using Euclidean distance</li>
    <li>Computes <strong>weights</strong> emphasizing closer matches</li>
    <li>Takes a <strong>weighted average</strong> of what actually happened in those situations</li>
</ol>

<p><strong>Result:</strong> You see what likely would have happened based on evidence from similar real-world scenarios.</p>
</div>
""", unsafe_allow_html=True)

st.write("")

# ============================================================================
# SIDEBAR - INPUT CONTROLS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Scenario Configuration")
    st.caption("🔄 Results update automatically as you change parameters")
    
    st.markdown("---")
    
    # Population and forecast
    st.subheader("📊 Analysis Settings")
    population = st.number_input(
        "Population",
        min_value=100000,
        max_value=1500000000,
        value=10000000,
        step=1000000,
        help="Total population size"
    )
    
    forecast_weeks = st.slider(
        "Forecast Horizon (weeks)",
        min_value=4,
        max_value=16,
        value=12,
        help="Number of weeks to project forward"
    )
    
    st.markdown("---")
    
    # Country Profile
    st.subheader("🌍 Country Profile")
    
    wealth_index = st.slider(
        "Wealth Index",
        min_value=0.0,
        max_value=1.0,
        value=0.6,
        step=0.05,
        help="Economic development level (0=Low, 1=High)"
    )
    
    health_index = st.slider(
        "Health System Index",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help="Healthcare capacity (0=Weak, 1=Strong)"
    )
    
    education_index = st.slider(
        "Education Index",
        min_value=0.0,
        max_value=1.0,
        value=0.65,
        step=0.05,
        help="Education level (0=Low, 1=High)"
    )
    
    st.markdown("---")
    
    # Vaccination and Economic Support
    st.subheader("💉 Vaccination & Support")
    
    vaccination_pct = st.slider(
        "Vaccination Coverage (%)",
        min_value=0,
        max_value=100,
        value=30,
        step=5,
        help="Percentage of population vaccinated"
    )
    
    income_support_level = st.select_slider(
        "Income Support Level",
        options=["None", "Partial", "Full"],
        value="Partial",
        help="Government economic support for affected individuals"
    )
    
    st.markdown("---")
    
    # NPIs
    st.subheader("🛡️ Non-Pharmaceutical Interventions")
    st.caption("0 = None, 1 = Maximum")
    
    with st.expander("School & Work"):
        npi_1 = st.slider("School Closures", 0.0, 1.0, 0.5, 0.1, key="npi_1")
        npi_2 = st.slider("Workplace Closures", 0.0, 1.0, 0.4, 0.1, key="npi_2")
    
    with st.expander("Public Activities"):
        npi_3 = st.slider("Cancel Public Events", 0.0, 1.0, 0.6, 0.1, key="npi_3")
        npi_4 = st.slider("Restrictions on Gatherings", 0.0, 1.0, 0.5, 0.1, key="npi_4")
    
    with st.expander("Movement & Travel"):
        npi_5 = st.slider("Public Transport Restrictions", 0.0, 1.0, 0.3, 0.1, key="npi_5")
        npi_6 = st.slider("Stay-at-Home Requirements", 0.0, 1.0, 0.4, 0.1, key="npi_6")
        npi_7 = st.slider("Internal Movement Restrictions", 0.0, 1.0, 0.3, 0.1, key="npi_7")
        npi_8 = st.slider("International Travel Controls", 0.0, 1.0, 0.7, 0.1, key="npi_8")
    
    with st.expander("Communication"):
        npi_9 = st.slider("Public Information Campaigns", 0.0, 1.0, 0.8, 0.1, key="npi_9")
    
    st.markdown("---")
    
    # Advanced settings
    st.subheader("Others")

    with st.expander("🔧 Advanced Settings"):
        k_neighbours = st.slider(
            "K Nearest Neighbours",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="Number of similar historical episodes to match"
        )
        
        epsilon = st.slider(
            "Distance Epsilon",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            step=0.001,
            format="%.3f",
            help="Small value to avoid division by zero in weighting"
        )

# ============================================================================
# MAIN CONTENT - COMPUTE AND DISPLAY RESULTS
# ============================================================================

# Convert inputs to feature vector
npis = [npi_1, npi_2, npi_3, npi_4, npi_5, npi_6, npi_7, npi_8, npi_9]
vacc_share = vaccination_pct / 100
income_support_map = {"None": 0, "Partial": 0.5, "Full": 1}
income_support = income_support_map[income_support_level]

# Build scenario vector
scenario_vector = build_feature_vector(
    wealth_index, health_index, education_index,
    npis, vacc_share, income_support
)

# Show loading state
with st.spinner("🔍 Finding similar situations and generating projections..."):
    # Find nearest neighbours
    nearest_neighbours = find_nearest_neighbours(
        scenario_vector,
        st.session_state.historical_data,
        k=k_neighbours
    )
    
    # Compute weights
    weights = compute_weights(nearest_neighbours, epsilon=epsilon)
    
    # Generate projections
    projections = generate_projections(
        nearest_neighbours,
        weights,
        forecast_weeks,
        population
    )
    
    # Calculate stringency
    stringency = calculate_stringency(npis)
    
    # Store results
    results = {
        'scenario_vector': scenario_vector,
        'nearest_neighbours': nearest_neighbours,
        'weights': weights,
        'projections': projections,
        'stringency': stringency
    }

# ============================================================================
# DISPLAY RESULTS IN TABS
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Projections",
    "🎯 Similar Situations",
    "📋 Scenario Details",
    "📖 Methodology"
])

# -------------------------
# TAB 1: PROJECTIONS
# -------------------------
with tab1:
    st.subheader("Outcome Projections")
    st.caption(f"Based on weighted average of {k_neighbours} most similar historical situations")
    
    st.write("")
    
    # Main layout: Left side (metrics + compliance) and Right side (weekly case chart) - equal sized
    left_col, right_col = st.columns(2)
    
    with left_col:
        # Add a summary bar
        st.markdown("""
        <div style='text-align: center; padding: 6px; background-color: #1f77b4; border-radius: 10px; margin-bottom: 15px;'>
            <h3 style='margin: 0; color: white;'>Summary</h3>
        </div>
        """, unsafe_allow_html=True)
        # 2x2 Metrics matrix with green text
        metric_row1_col1, metric_row1_col2 = st.columns(2)
        
        total_cases = int(np.sum(projections['cases_absolute']))
        total_deaths = int(np.sum(projections['deaths_absolute']))
        avg_mortality = np.mean(projections['mortality']) * 100
        
        with metric_row1_col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #FFFFFF; border-radius: 10px;'>
                <div style='color: green; font-size: 20px; font-weight: 500;'>Total Cases</div>
                <div style='color: green; font-size: 32px; font-weight: bold; margin-top: 10px;'>{total_cases:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_row1_col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #FFFFFF; border-radius: 10px;'>
                <div style='color: green; font-size: 20px; font-weight: 500;'>Total Deaths</div>
                <div style='color: green; font-size: 32px; font-weight: bold; margin-top: 10px;'>{total_deaths:,}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        metric_row2_col1, metric_row2_col2 = st.columns(2)
        
        with metric_row2_col1:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #FFFFFF; border-radius: 10px;'>
                <div style='color: green; font-size: 20px; font-weight: 500;'>Avg Mortality Rate</div>
                <div style='color: green; font-size: 32px; font-weight: bold; margin-top: 10px;'>{avg_mortality:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
        
        with metric_row2_col2:
            st.markdown(f"""
            <div style='text-align: center; padding: 20px; background-color: #FFFFFF; border-radius: 10px;'>
                <div style='color: green; font-size: 20px; font-weight: 500;'>Stringency Index</div>
                <div style='color: green; font-size: 32px; font-weight: bold; margin-top: 10px;'>{stringency:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.write("")
        
        # Compliance bar below metrics matrix - same width as matrix
        compliance_pct = projections['compliance'] * 100
        st.info(f"**Expected Public Compliance: {compliance_pct:.1f}%**")
    
    with right_col:
        # Weekly case chart - increased height to match metrics + compliance bar
        st.markdown("""
        <div style='text-align: center; padding: 6px; background-color: #DEE1E2; border: 2px solid #dee2e6; border-radius: 10px; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: #333;'>Weekly Case Chart</h4>
        </div>
        """, unsafe_allow_html=True)
        
        weeks = list(range(1, forecast_weeks + 1))
        
        fig_cases = go.Figure()
        fig_cases.add_trace(go.Scatter(
            x=weeks,
            y=projections['cases_absolute'],
            mode='lines+markers',
            name='Cases',
            line=dict(color='blue', width=3),
            marker=dict(size=6)
        ))
        fig_cases.update_layout(
            height=360,
            xaxis_title="Week",
            yaxis_title="Number of Cases",
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            margin=dict(t=20, b=40, l=50, r=20),
            # Add border to the chart
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            )
        )
        st.plotly_chart(fig_cases, use_container_width=True)
    
    st.write("")
    st.write("")
    
    # Bottom row: Weekly death chart and Mortality rate chart - reduced size
    chart_col1, chart_col2 = st.columns(2)
    
    weeks = list(range(1, forecast_weeks + 1))
    
    with chart_col1:
        st.markdown("""
        <div style='text-align: center; padding: 6px; background-color: #DEE1E2; border: 2px solid #dee2e6; border-radius: 10px; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: #333;'>Weekly Death Chart</h4>
        </div>
        """, unsafe_allow_html=True)
        
        fig_deaths = go.Figure()
        fig_deaths.add_trace(go.Scatter(
            x=weeks,
            y=projections['deaths_absolute'],
            mode='lines+markers',
            name='Deaths',
            line=dict(color='red', width=3),
            marker=dict(size=6)
        ))
        fig_deaths.update_layout(
            height=280,
            xaxis_title="Week",
            yaxis_title="Number of Deaths",
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            margin=dict(t=20, b=40, l=50, r=20),
            # Add border to the chart
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            )
        )
        st.plotly_chart(fig_deaths, use_container_width=True)
    
    with chart_col2:
        st.markdown("""
        <div style='text-align: center; padding: 6px; background-color: #DEE1E2; border: 2px solid #dee2e6; border-radius: 10px; margin-bottom: 10px;'>
            <h4 style='margin: 0; color: #333;'>Mortality Rate Chart</h4>
        </div>
        """, unsafe_allow_html=True)
        
        fig_mortality = go.Figure()
        fig_mortality.add_trace(go.Scatter(
            x=weeks,
            y=projections['mortality'] * 100,
            mode='lines+markers',
            name='Mortality Rate',
            line=dict(color='purple', width=3),
            marker=dict(size=6)
        ))
        fig_mortality.update_layout(
            height=280,
            xaxis_title="Week",
            yaxis_title="Mortality Rate (%)",
            plot_bgcolor='white',
            paper_bgcolor='white',
            hovermode='x unified',
            margin=dict(t=20, b=40, l=50, r=20),
            # Add border to the chart
            xaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            ),
            yaxis=dict(
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            )
        )
        st.plotly_chart(fig_mortality, use_container_width=True)

# -------------------------
# TAB 2: SIMILAR SITUATIONS
# -------------------------
with tab2:
    st.subheader("Most Similar Historical Situations")
    st.caption(f"Showing top {min(5, k_neighbours)} nearest neighbours")
    
    # Create table of nearest neighbours
    neighbours_data = []
    for i, (distance, episode) in enumerate(results['nearest_neighbours'][:5]):
        neighbours_data.append({
            'Rank': i + 1,
            'Country': episode['country'],
            'Week': episode['week'],
            'Distance': f"{distance:.4f}",
            'Weight': f"{results['weights'][i]:.4f}",
            'Compliance': f"{episode['compliance']*100:.1f}%",
            'Wealth': f"{episode['wealth']:.2f}",
            'Health': f"{episode['health']:.2f}",
            'Education': f"{episode['education']:.2f}",
            'Vaccination': f"{episode['vacc_share']*100:.0f}%"
        })
    
    df_neighbours = pd.DataFrame(neighbours_data)
    st.dataframe(df_neighbours, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Distance distribution
    st.subheader("Distance Distribution")
    st.caption("How similar are the matched situations?")
    
    all_distances = [d for d, _ in results['nearest_neighbours']]
    
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=all_distances,
        nbinsx=20,
        marker_color='lightblue',
        marker_line_color='blue',
        marker_line_width=1
    ))
    
    fig_dist.update_layout(
        height=240,
        xaxis_title="Distance from Scenario",
        yaxis_title="Count",
        plot_bgcolor='white',
        showlegend=False
    )
    
    st.plotly_chart(fig_dist, use_container_width=True)
    
    st.info(f"""
    💡 **Interpretation:**
    - Average distance: {np.mean(all_distances):.4f}
    - Minimum distance: {np.min(all_distances):.4f}
    - Maximum distance: {np.max(all_distances):.4f}
    
    Smaller distances indicate more similar historical situations. 
    The weighted average emphasizes closer matches.
    """)

# -------------------------
# TAB 3: SCENARIO DETAILS
# -------------------------
with tab3:
    st.subheader("📋 Your Scenario Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Country Profile:**")
        profile_data = {
            "Component": ["Wealth Index", "Health Index", "Education Index"],
            "Value": [f"{wealth_index:.1f}", f"{health_index:.1f}", f"{education_index:.1f}"]
        }
        st.table(pd.DataFrame(profile_data))
        
        st.write("**Economic Support:**")
        st.write(f"- Vaccination Coverage: {vaccination_pct}%")
        st.write(f"- Income Support: {income_support_level}")
    
    with col2:
        st.write("**Non-Pharmaceutical Interventions:**")
        npi_data = {
            "Intervention": [
                "School Closures", "Workplace Closures", "Public Events",
                "Gathering Limits", "Public Transport", "Stay-at-Home",
                "Internal Movement", "International Travel", "Information Campaigns"
            ],
            "Level": [
                npi_1, npi_2, npi_3, npi_4, npi_5, 
                npi_6, npi_7, npi_8, npi_9
            ],
            "Value": [f"{v:.2f}" for v in npis]
        }
        st.dataframe(pd.DataFrame(npi_data), hide_index=True, use_container_width=True)
    
    st.markdown("---")
    
    st.write("**Feature Vector (14 dimensions):**")
    st.code(f"{results['scenario_vector']}", language=None)
    
    st.markdown("---")
    
    st.write("**Analysis Parameters:**")
    st.write(f"- Population: {population:,}")
    st.write(f"- Forecast Horizon: {forecast_weeks} weeks")
    st.write(f"- Nearest Neighbours (K): {k_neighbours}")
    st.write(f"- Distance Epsilon: {epsilon}")

# -------------------------
# TAB 4: METHODOLOGY
# -------------------------
with tab4:
    st.subheader("📖 Methodology")
    
    st.markdown("""
    ### How the Evidence Explorer Works
    
    #### 1️⃣ Feature Vector Construction
    Your scenario is converted into a **14-dimensional feature vector**:
```
    x* = [wealth, health, education, npi_1, ..., npi_9, vacc_share, income_support]
```
    
    All values are normalized to [0, 1] for meaningful comparisons.
    
    #### 2️⃣ Nearest Neighbour Search
    We compare your scenario to thousands of real historical (country, week) episodes using **Euclidean distance**:
```
    distance = √(Σ(x_i - x*)²)
```
    
    #### 3️⃣ Weighting
    Closer matches receive higher weights:
```
    w_i = 1 / (distance_i + ε)
    w_i = w_i / Σw_i  (normalized)
```
    
    #### 4️⃣ Trajectory Averaging
    Projections are **weighted averages** of what actually happened in similar situations:
```
    cases[week] = Σ(w_i × cases_i[week])
    deaths[week] = Σ(w_i × deaths_i[week])
```
    
    #### 5️⃣ Derived Metrics
    - **Mortality Rate**: deaths / (cases + ε)
    - **Stringency**: (1/9) × Σ(100 × npi_j)
    - **Compliance**: Weighted average from matched episodes
    
    #### 6️⃣ Scaling
    Per-capita projections are scaled to your selected population:
```
    absolute_cases = cases_per_100k × (population / 100,000)
```
    
    ---
    
    ### Why This Approach?
    
    ✅ **Transparent**: All projections trace back to real historical data
    
    ✅ **No Black Box**: No complex simulation assumptions
    
    ✅ **Evidence-Based**: Grounded in what actually happened
    
    ✅ **Interpretable**: You can inspect the similar situations used
    
    ⚠️ **Limitation**: Quality depends on having similar historical situations in the database
    
    ---
    
    ### Data Sources
    - Oxford COVID-19 Government Response Tracker (OxCGRT)
    - Our World in Data
    - Google Mobility Reports
    - WHO COVID-19 Database
    
    *Note: This demo uses simulated data for illustration. Production version would use actual historical database.*
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

st.caption(f"Copyright {datetime.now().strftime('%Y')} WorldQuant University. This content is licensed solely for personal use. Redistribution or publication of this material is strictly prohibited.")
