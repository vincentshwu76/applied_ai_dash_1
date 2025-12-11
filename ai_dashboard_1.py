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
    .main-header {
        font-size: 10rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .info-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 10px 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
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

if 'results' not in st.session_state:
    st.session_state.results = None

# ============================================================================
# HEADER
# ============================================================================
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("# COVID Evidence Explorer")
    st.caption("Data-driven projections based on historical evidence from similar situations")
st.write("")
st.write("")
st.write("")

_, exp_col, _ = st.columns([1,3,1])
with exp_col:
    with st.expander("**📖 Introduction to This Dashboard**"):

            st.markdown("""
    <div class="info-box">
    <h3>📖 How This Dashboard Works</h3>
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
    
    <p>We compare this profile to <strong>thousands of real country-weeks</strong> recorded in global COVID datasets.</p>
    
    <p>The system finds the <strong>closest matches</strong> ("nearest neighbours") and looks at what happened next in those situations.</p>
    
    <p>The curves you see (cases, deaths, mortality) are created by <strong>averaging the actual trajectories</strong> of those similar historical episodes.</p>
    
    <p><strong>Compliance (%)</strong> is also learned from these neighbours. It reflects how populations in similar real situations responded to government policy.</p>
    
    <p><strong>Stringency</strong> is calculated directly from the NPIs you select using the Oxford Government Response Tracker method.</p>
    
    <p>✅ This approach provides <strong>transparent, data-driven insights</strong> without relying on mechanistic epidemiological models.</p>
    
    <p>✅ All projections are <strong>grounded in what actually happened</strong> in countries with similar characteristics and policies.</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ============================================================================
# SIDEBAR - USER INPUTS
# ============================================================================
with st.sidebar:
    st.title("Scenario Configuration")
    
    # -------------------------
    # A. COUNTRY PROFILE (EXPANDABLE)
    # -------------------------
    with st.expander("📊 Country Profile (HDI Components)", expanded=False):
        st.caption("These describe the long-term developmental baseline")
        
        wealth_index = st.slider(
            "💰 Wealth Index (HDI Income)",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0 = very low income; 1 = very high income"
        )
        #st.progress(wealth_index)
        
        health_index = st.slider(
            "🏥 Health Index (Life Expectancy)",
            min_value=0.0,
            max_value=1.0,
            value=0.8,
            step=0.1,
            help="0 = low life expectancy; 1 = high life expectancy"
        )
        #st.progress(health_index)
        
        education_index = st.slider(
            "🎓 Education Index",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="0 = low schooling/literacy; 1 = high"
        )
        #st.progress(education_index)
    
    # -------------------------
    # B. POLICY LEVERS (EXPANDABLE)
    # -------------------------
    with st.expander("🎚️ Policy Levers", expanded=False):
        
        # Vaccination
        vaccination_pct = st.slider(
            "💉 Vaccination Coverage (%)",
            min_value=0,
            max_value=100,
            value=60,
            step=10,
            help="Percentage of population vaccinated"
        )
        vacc_share = vaccination_pct / 100.0
        #st.progress(vacc_share)
        
        # Income Support
        income_support_level = st.select_slider(
            "💵 Income Support",
            options=["None", "Limited", "High"],
            value="Limited",
            help="Level of government economic support"
        )
        income_support_map = {"None": 0.0, "Limited": 0.5, "High": 1.0}
        income_support = income_support_map[income_support_level]
    
    # -------------------------
    # C. NPIs (9 policies) - EXPANDABLE
    # -------------------------
    with st.expander("🚦 Non-Pharmaceutical Interventions (NPIs)", expanded=False):
        st.caption("Select the intensity of each intervention")
        
        npi_1 = st.select_slider(
            "🏫 School Closures",
            options=["Open", "Partial", "Strong"],
            value="Partial"
        ) 
        npi_1_map = {"Open": 0.0, "Partial": 0.5, "Strong": 1.0}
        npi_1_value = npi_1_map[npi_1]
        
        npi_2 = st.select_slider(
            "🏢 Workplace Closures",
            options=["Open", "Partial", "Strong"],
            value="Partial"
        )
        npi_2_map = {"Open": 0.0, "Partial": 0.5, "Strong": 1.0}
        npi_2_value = npi_2_map[npi_2]
        
        npi_3 = st.select_slider(
            "🎭 Public Events",
            options=["Allowed", "Restricted", "Banned"],
            value="Restricted"
        )
        npi_3_map = {"Allowed": 0.0, "Restricted": 0.5, "Banned": 1.0}
        npi_3_value = npi_3_map[npi_3]
        
        npi_4 = st.select_slider(
            "👥 Gathering Limits",
            options=["No Limit", "Limits", "Strict Limits"],
            value="Limits"
        )
        npi_4_map = {"No Limit": 0.0, "Limits": 0.5, "Strict Limits": 1.0}
        npi_4_value = npi_4_map[npi_4]
        
        npi_5 = st.select_slider(
            "🚌 Public Transport",
            options=["Open", "Restricted", "Closed"],
            value="Open"
        )
        npi_5_map = {"Open": 0.0, "Restricted": 0.5, "Closed": 1.0}
        npi_5_value = npi_5_map[npi_5]
        
        npi_6 = st.select_slider(
            "🏠 Stay-at-Home Orders",
            options=["None", "Partial", "Required"],
            value="None"
        )
        npi_6_map = {"None": 0.0, "Partial": 0.5, "Required": 1.0}
        npi_6_value = npi_6_map[npi_6]
        
        npi_7 = st.select_slider(
            "🚗 Internal Movement",
            options=["Free", "Restricted", "Strong"],
            value="Free"
        )
        npi_7_map = {"Free": 0.0, "Restricted": 0.5, "Strong": 1.0}
        npi_7_value = npi_7_map[npi_7]
        
        npi_8 = st.select_slider(
            "✈️ International Travel",
            options=["Open", "Screening", "Quarantine", "Closed"],
            value="Screening"
        )
        npi_8_map = {"Open": 0.0, "Screening": 0.33, "Quarantine": 0.67, "Closed": 1.0}
        npi_8_value = npi_8_map[npi_8]
        
        npi_9 = st.select_slider(
            "📢 Public Information Campaigns",
            options=["None", "Limited", "Extensive"],
            value="Extensive"
        )
        npi_9_map = {"None": 0.0, "Limited": 0.5, "Extensive": 1.0}
        npi_9_value = npi_9_map[npi_9]
        
        # Collect all NPIs
        npis = [npi_1_value, npi_2_value, npi_3_value, npi_4_value, npi_5_value, 
                npi_6_value, npi_7_value, npi_8_value, npi_9_value]
    
    # -------------------------
    # D. POPULATION & FORECAST (EXPANDABLE)
    # -------------------------
    with st.expander("👥 Population & Forecast", expanded=False):
        st.caption("For scaling outputs only (not part of matching)")
        
        population = st.number_input(
            "Population Size",
            min_value=100000,
            max_value=1500000000,
            value=10000000,
            step=1000000,
            format="%d"
        )
        
        st.markdown("---")
        
        forecast_weeks = st.slider(
            "📅 Weeks to Project",
            min_value=4,
            max_value=16,
            value=8,
            step=1
        )
    
    # -------------------------
    # E. ADVANCED SETTINGS (EXPANDABLE - COLLAPSED BY DEFAULT)
    # -------------------------
    with st.expander("⚙️ Advanced Settings", expanded=False):
        k_neighbours = st.slider(
            "Number of Nearest Neighbours (K)",
            min_value=10,
            max_value=100,
            value=30,
            step=5,
            help="How many similar historical situations to average"
        )
        
        epsilon = st.number_input(
            "Distance Epsilon",
            min_value=0.001,
            max_value=0.1,
            value=0.01,
            format="%.3f",
            help="Small value to avoid division by zero in weights"
        )
    
    st.markdown("---")
           
    # -------------------------
    # RUN BUTTON
    # -------------------------
    run_analysis = st.button("🔍 Find Similar Situations & Project", type="primary", use_container_width=True)

# ============================================================================
# MAIN CONTENT - PROCESS AND DISPLAY RESULTS
# ============================================================================

if run_analysis:
    with st.spinner("🔄 Searching for similar historical situations..."):
        # Step 1: Build feature vector
        scenario_vector = build_feature_vector(
            wealth_index, health_index, education_index,
            npis, vacc_share, income_support
        )
        
        # Step 2: Find nearest neighbours
        nearest_neighbours = find_nearest_neighbours(
            scenario_vector, 
            st.session_state.historical_data, 
            k=k_neighbours
        )
        
        # Step 3: Compute weights
        weights = compute_weights(nearest_neighbours, epsilon=epsilon)
        
        # Step 4: Generate projections
        projections = generate_projections(
            nearest_neighbours, weights, forecast_weeks, population
        )
        
        # Step 5: Calculate stringency
        stringency = calculate_stringency(npis)
        
        # Store results
        st.session_state.results = {
            'projections': projections,
            'stringency': stringency,
            'nearest_neighbours': nearest_neighbours,
            'weights': weights,
            'scenario_vector': scenario_vector
        }
    
    st.success("✅ Analysis complete! Scroll down to see results.")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

if st.session_state.results is not None:
    results = st.session_state.results
    projections = results['projections']
    stringency = results['stringency']
    compliance = projections['compliance']
    
    st.markdown("---")
    st.header("📊 Projection Results")
    st.caption(f"Based on {k_neighbours} similar historical situations")
    
    # -------------------------
    # KEY METRICS
    # -------------------------
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_cases = int(projections['cases_absolute'].sum())
        st.metric(
            "Total Projected Cases",
            f"{total_cases:,}",
            help=f"Over {forecast_weeks} weeks"
        )
    
    with col2:
        total_deaths = int(projections['deaths_absolute'].sum())
        st.metric(
            "Total Projected Deaths",
            f"{total_deaths:,}",
            help=f"Over {forecast_weeks} weeks"
        )
    
    with col3:
        st.metric(
            "Stringency Index",
            f"{stringency:.1f}",
            help="Based on selected NPIs (0-100)"
        )
    
    with col4:
        st.metric(
            "Expected Compliance",
            f"{compliance * 100:.1f}%",
            help="Learned from similar historical situations"
        )
    
    st.markdown("---")
    
    # -------------------------
    # TABS FOR DIFFERENT VIEWS
    # -------------------------
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Projections", "🎯 Similar Situations", "📋 Scenario Details", "ℹ️ Methodology"])
    
    # -------------------------
    # TAB 1: PROJECTIONS
    # -------------------------
    with tab1:
        # Generate week labels
        weeks = list(range(1, forecast_weeks + 1))
        
        # ========================================================================
        # CHART 1: CASES (STYLED)
        # ========================================================================
        st.subheader("Projected Cases")
        
        fig_cases = go.Figure()
        
        fig_cases.add_trace(go.Scatter(
            x=weeks,
            y=projections['cases_absolute'],
            mode='lines+markers',
            name='Absolute Cases',
            line=dict(color='#1f77b4', width=4, shape='spline'),
            marker=dict(
                size=8,
                color='#1f77b4',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Week %{x}</b><br>Cases: %{y:,.0f}<extra></extra>'
        ))
        
        fig_cases.update_layout(
            height=450,
            xaxis_title="Week",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True,
                tickformat=','
            ),
            margin=dict(l=70, r=40, t=40, b=60)
        )
        
        st.plotly_chart(fig_cases, use_container_width=True)
        
        # ========================================================================
        # CHART 2: DEATHS (STYLED)
        # ========================================================================
        st.markdown("---")
        st.subheader("Projected Deaths")
        
        fig_deaths = go.Figure()
        
        fig_deaths.add_trace(go.Scatter(
            x=weeks,
            y=projections['deaths_absolute'],
            mode='lines+markers',
            name='Absolute Deaths',
            line=dict(color='#e74c3c', width=4, shape='spline'),
            marker=dict(
                size=8,
                color='#e74c3c',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Week %{x}</b><br>Deaths: %{y:,.0f}<extra></extra>'
        ))
        
        fig_deaths.update_layout(
            height=450,
            xaxis_title="Week",
            yaxis_title="Number of Deaths",
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True,
                tickformat=','
            ),
            margin=dict(l=70, r=40, t=40, b=60)
        )
        
        st.plotly_chart(fig_deaths, use_container_width=True)
        
        # ========================================================================
        # CHART 3: MORTALITY RATE (STYLED)
        # ========================================================================
        st.markdown("---")
        st.subheader("Mortality Rate")
        
        fig_mortality = go.Figure()
        
        fig_mortality.add_trace(go.Scatter(
            x=weeks,
            y=projections['mortality'] * 100,  # Convert to percentage
            mode='lines+markers',
            name='Mortality Rate (%)',
            line=dict(color='#f39c12', width=4, shape='spline'),
            marker=dict(
                size=8,
                color='#f39c12',
                line=dict(color='white', width=2)
            ),
            hovertemplate='<b>Week %{x}</b><br>Mortality: %{y:.2f}%<extra></extra>'
        ))
        
        fig_mortality.update_layout(
            height=450,
            xaxis_title="Week",
            yaxis_title="Mortality Rate (%)",
            hovermode='x unified',
            plot_bgcolor='rgba(248, 249, 250, 1)',
            paper_bgcolor='white',
            font=dict(family="Arial, sans-serif", size=12, color="#2c3e50"),
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(200, 200, 200, 0.3)',
                showline=True,
                linewidth=2,
                linecolor='#dee2e6',
                mirror=True,
                ticksuffix='%'
            ),
            margin=dict(l=70, r=40, t=40, b=60)
        )
        
        st.plotly_chart(fig_mortality, use_container_width=True)
        
        # ========================================================================
        # PER-CAPITA VIEW (STYLED)
        # ========================================================================
        st.markdown("---")
        st.subheader("Per 100,000 Population")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_cases_pc = go.Figure()
            fig_cases_pc.add_trace(go.Scatter(
                x=weeks,
                y=projections['cases_per_100k'],
                mode='lines+markers',
                name='Cases per 100k',
                line=dict(color='#3498db', width=3, shape='spline'),
                marker=dict(size=6, color='#3498db', line=dict(color='white', width=1)),
                hovertemplate='<b>Week %{x}</b><br>Cases: %{y:.1f}<extra></extra>'
            ))
            fig_cases_pc.update_layout(
                title="Cases per 100k",
                height=320,
                xaxis_title="Week",
                yaxis_title="Cases per 100k",
                plot_bgcolor='rgba(248, 249, 250, 1)',
                paper_bgcolor='white',
                font=dict(size=11, color="#2c3e50"),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showline=True,
                    linecolor='#dee2e6'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showline=True,
                    linecolor='#dee2e6'
                ),
                margin=dict(l=60, r=30, t=50, b=50)
            )
            st.plotly_chart(fig_cases_pc, use_container_width=True)
        
        with col2:
            fig_deaths_pc = go.Figure()
            fig_deaths_pc.add_trace(go.Scatter(
                x=weeks,
                y=projections['deaths_per_100k'],
                mode='lines+markers',
                name='Deaths per 100k',
                line=dict(color='#e74c3c', width=3, shape='spline'),
                marker=dict(size=6, color='#e74c3c', line=dict(color='white', width=1)),
                hovertemplate='<b>Week %{x}</b><br>Deaths: %{y:.2f}<extra></extra>'
            ))
            fig_deaths_pc.update_layout(
                title="Deaths per 100k",
                height=320,
                xaxis_title="Week",
                yaxis_title="Deaths per 100k",
                plot_bgcolor='rgba(248, 249, 250, 1)',
                paper_bgcolor='white',
                font=dict(size=11, color="#2c3e50"),
                xaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showline=True,
                    linecolor='#dee2e6'
                ),
                yaxis=dict(
                    showgrid=True,
                    gridcolor='rgba(200, 200, 200, 0.3)',
                    showline=True,
                    linecolor='#dee2e6'
                ),
                margin=dict(l=60, r=30, t=50, b=50)
            )
            st.plotly_chart(fig_deaths_pc, use_container_width=True)



    # with tab1:
    #     # Generate week labels
    #     weeks = list(range(1, forecast_weeks + 1))
        
    #     # Cases Chart
    #     st.subheader("Projected Cases")
        
    #     fig_cases = go.Figure()
        
    #     fig_cases.add_trace(go.Scatter(
    #         x=weeks,
    #         y=projections['cases_absolute'],
    #         mode='lines+markers',
    #         name='Absolute Cases',
    #         line=dict(color='blue', width=3),
    #         hovertemplate='<b>Week %{x}</b><br>Cases: %{y:,.0f}<extra></extra>'
    #     ))
        
    #     fig_cases.update_layout(
    #         height=400,
    #         xaxis_title="Week",
    #         yaxis_title="Number of Cases",
    #         hovermode='x unified',
    #         plot_bgcolor='white',
    #         xaxis=dict(showgrid=True, gridcolor='lightgray'),
    #         yaxis=dict(showgrid=True, gridcolor='lightgray')
    #     )
        
    #     st.plotly_chart(fig_cases, use_container_width=True)
        
    #     # Deaths Chart
    #     st.markdown("---")
    #     st.subheader("Projected Deaths")
        
    #     fig_deaths = go.Figure()
        
    #     fig_deaths.add_trace(go.Scatter(
    #         x=weeks,
    #         y=projections['deaths_absolute'],
    #         mode='lines+markers',
    #         name='Absolute Deaths',
    #         line=dict(color='red', width=3),
    #         hovertemplate='<b>Week %{x}</b><br>Deaths: %{y:,.0f}<extra></extra>'
    #     ))
        
    #     fig_deaths.update_layout(
    #         height=400,
    #         xaxis_title="Week",
    #         yaxis_title="Number of Deaths",
    #         hovermode='x unified',
    #         plot_bgcolor='white',
    #         xaxis=dict(showgrid=True, gridcolor='lightgray'),
    #         yaxis=dict(showgrid=True, gridcolor='lightgray')
    #     )
        
    #     st.plotly_chart(fig_deaths, use_container_width=True)
        
    #     # Mortality Rate Chart
    #     st.markdown("---")
    #     st.subheader("Mortality Rate")
        
    #     fig_mortality = go.Figure()
        
    #     fig_mortality.add_trace(go.Scatter(
    #         x=weeks,
    #         y=projections['mortality'] * 100,  # Convert to percentage
    #         mode='lines+markers',
    #         name='Mortality Rate (%)',
    #         line=dict(color='orange', width=3),
    #         hovertemplate='<b>Week %{x}</b><br>Mortality: %{y:.2f}%<extra></extra>'
    #     ))
        
    #     fig_mortality.update_layout(
    #         height=400,
    #         xaxis_title="Week",
    #         yaxis_title="Mortality Rate (%)",
    #         hovermode='x unified',
    #         plot_bgcolor='white',
    #         xaxis=dict(showgrid=True, gridcolor='lightgray'),
    #         yaxis=dict(showgrid=True, gridcolor='lightgray')
    #     )
        
    #     st.plotly_chart(fig_mortality, use_container_width=True)
        
    #     # Per-capita view
    #     st.markdown("---")
    #     st.subheader("Per 100,000 Population")
        
    #     col1, col2 = st.columns(2)
        
    #     with col1:
    #         fig_cases_pc = go.Figure()
    #         fig_cases_pc.add_trace(go.Scatter(
    #             x=weeks,
    #             y=projections['cases_per_100k'],
    #             mode='lines+markers',
    #             name='Cases per 100k',
    #             line=dict(color='blue', width=2)
    #         ))
    #         fig_cases_pc.update_layout(
    #             title="Cases per 100k",
    #             height=300,
    #             xaxis_title="Week",
    #             yaxis_title="Cases per 100k",
    #             plot_bgcolor='white'
    #         )
    #         st.plotly_chart(fig_cases_pc, use_container_width=True)
        
    #     with col2:
    #         fig_deaths_pc = go.Figure()
    #         fig_deaths_pc.add_trace(go.Scatter(
    #             x=weeks,
    #             y=projections['deaths_per_100k'],
    #             mode='lines+markers',
    #             name='Deaths per 100k',
    #             line=dict(color='red', width=2)
    #         ))
    #         fig_deaths_pc.update_layout(
    #             title="Deaths per 100k",
    #             height=300,
    #             xaxis_title="Week",
    #             yaxis_title="Deaths per 100k",
    #             plot_bgcolor='white'
    #         )
    #         st.plotly_chart(fig_deaths_pc, use_container_width=True)
    
    # -------------------------
    # TAB 2: SIMILAR SITUATIONS
    # -------------------------
    with tab2:
        st.subheader("🎯 Most Similar Historical Situations")
        st.caption(f"Showing top {min(10, k_neighbours)} nearest neighbours")
        
        # Create table of nearest neighbours
        neighbours_data = []
        for i, (distance, episode) in enumerate(results['nearest_neighbours'][:10]):
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
        st.subheader("📊 Distance Distribution")
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
            height=300,
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

else:
    # Show instructions when no results yet
    st.info("""
    👈 **Get Started:**
    
    1. Configure your hypothetical country's characteristics in the sidebar
    2. Set vaccination levels and economic support
    3. Choose non-pharmaceutical interventions (NPIs)
    4. Set population and forecast horizon
    5. Click "Find Similar Situations & Project"
    
    The system will find real historical COVID situations that match your scenario and show you what happened next.
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")

st.caption(f"Copyright {datetime.now().strftime('%Y')} WorldQuant University. This content is licensed solely for personal use. Redistribution or publication of this material is strictly prohibited.")
