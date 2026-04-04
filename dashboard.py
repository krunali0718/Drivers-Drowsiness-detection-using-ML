import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="Driver Alert Dashboard", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR PREMIUM LOOK ---
st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .metric-box {
        background-color: #1E2127;
        padding: 20px; border-radius: 10px;
        text-align: center; border: 1px solid #333;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    .metric-title { color: #A0AEC0; font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    .metric-value { color: #FFFFFF; font-size: 32px; font-weight: bold; }
    .metric-value.red { color: #FF4A4A; }
    .metric-value.orange { color: #FFA500; }
    .metric-value.green { color: #00E676; }
</style>
""", unsafe_allow_html=True)

st.title("Advanced Driver Drowsiness Analytics")
st.markdown("Monitor real-time fatigue events, distraction warnings, and historical performance using our Hybrid CV + AI Model. **Data Source: SQLite Database**")

DB_FILE = "driver_fatigue_database.sqlite"

@st.cache_data(ttl=3) # Auto-refresh data every 3 seconds to keep it real-time
def load_data():
    if os.path.exists(DB_FILE):
        try:
            # Querying directly from SQLite relational database
            conn = sqlite3.connect(DB_FILE)
            df = pd.read_sql_query("SELECT * FROM alerts ORDER BY timestamp DESC", conn)
            conn.close()
            
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                # rename columns to match the old interface slightly for the visual pie-charts
                df = df.rename(columns={"alert_type": "Alert_Type", "severity": "Severity", "perclos_value": "PERCLOS_Value", "timestamp":"Timestamp"})
                return df
        except Exception as e:
            st.error(f"Error loading SQL database: {e}")
    return pd.DataFrame(columns=["id", "Timestamp", "Alert_Type", "Severity", "PERCLOS_Value"])

df = load_data()

# --- SIDEBAR ---
st.sidebar.header("Dashboard Controls")
if st.sidebar.button("Refresh DB Request"):
    st.cache_data.clear()

st.sidebar.markdown("---")
st.sidebar.markdown("### HOW TO RUN LIVE DETECTION")
st.sidebar.info("""
1. Open your terminal
2. Run: `python advanced_drowsiness_test.py`
3. The dashboard will automatically query the SQLite engine for new alerts!
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**AI Architecture Metrics**")
st.sidebar.markdown("- **Model:** XGBoost Classifier")
st.sidebar.markdown("- **Estimated Accuracy:** 90%+ (SMOTE)")
st.sidebar.markdown("- **Database Engine:** SQLite Relational DB")

# --- MAIN DASHBOARD CONTENT ---
if df.empty:
    st.warning("SQL Database Table is empty. Please run `advanced_drowsiness_test.py` to start logging fatigue events to the SQLite engine.")
else:
    total_alerts = len(df)
    critical_sleep = len(df[df["Alert_Type"] == "Critical Sleep"])
    distractions = len(df[df["Alert_Type"] == "Distraction_or_Phone"])
    stress = len(df[df["Alert_Type"] == "Quarreling / Stress"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-box"><div class="metric-title">SQL Queries Executed</div><div class="metric-value">{total_alerts}</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Critical Sleep</div><div class="metric-value red">{critical_sleep}</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Distractions</div><div class="metric-value orange">{distractions}</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-box"><div class="metric-title">Stress/Anger</div><div class="metric-value">{stress}</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Incident Breakdown")
        alert_counts = df['Alert_Type'].value_counts().reset_index()
        alert_counts.columns = ['Alert_Type', 'Count']
        fig = px.pie(alert_counts, values='Count', names='Alert_Type', hole=0.4, 
                     color_discrete_sequence=px.colors.sequential.RdBu)
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font_color="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Real-Time PERCLOS Trend")
        df_sorted = df.sort_values(by="Timestamp")
        fig_line = px.line(df_sorted, x='Timestamp', y='PERCLOS_Value', color='Alert_Type', markers=True,
                           template="plotly_dark")
        fig_line.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               xaxis_title="SQL Timestamp", yaxis_title="PERCLOS (%)")
        st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")
    st.subheader("SQLite Database Raw Records (Top 10)")
    st.dataframe(df.head(10), use_container_width=True)
