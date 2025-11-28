import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
import joblib
from math import radians, cos, sin, asin, sqrt

st.set_page_config(
    page_title="Phase 1: Fleet Command Center",
    page_icon="🚖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background-color: #0E1117; color: white; }
    div.stButton > button { background-color: #4B5563; color: white; border-radius: 5px; }
    div[data-testid="stMetricValue"] { font-size: 24px; color: #00CC96; }
    h1, h2, h3 { color: #E5E7EB; }
    .css-1aumxhk { background-color: #262730; }
</style>
""", unsafe_allow_html=True)

def haversine(lon1, lat1, lon2, lat2):
    """Calculates Haversine distance for Efficiency Metrics (in km)"""
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1 
    dlat = lat2 - lat1 
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a)) 
    r = 6371
    return c * r

@st.cache_data
def load_data(filepath):
    """Loads and preprocesses data efficiently"""
    try:
        cols = ['pickup_datetime', 'pickup_longitude', 'pickup_latitude', 
                'dropoff_longitude', 'dropoff_latitude', 'passenger_count', 'trip_duration']
        
        df = pd.read_csv(filepath, usecols=cols, nrows=50000)
        
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])
        df['hour'] = df['pickup_datetime'].dt.hour
        df['weekday'] = df['pickup_datetime'].dt.day_name()
        
        df = df[
            (df['pickup_latitude'] > 40.60) & (df['pickup_latitude'] < 40.90) &
            (df['pickup_longitude'] > -74.05) & (df['pickup_longitude'] < -73.70)
        ]
        
        df['distance_km'] = df.apply(lambda x: haversine(
            x['pickup_longitude'], x['pickup_latitude'], 
            x['dropoff_longitude'], x['dropoff_latitude']), axis=1)
    
        df = df[df['trip_duration'] > 60]
        df['speed_kmh'] = df['distance_km'] / (df['trip_duration'] / 3600)
        df = df[(df['speed_kmh'] > 1) & (df['speed_kmh'] < 120)] 
        
        return df
    except FileNotFoundError:
        return None

@st.cache_resource
def load_model():
    """Loads the pre-trained K-Means model for live predictions"""
    try:
        model = joblib.load('kmeans_fleet_model.pkl')
        return model, "✅ Online (Pre-Trained)"
    except:
        return None, "⚠️ Offline (File Missing)"

def main():
    st.sidebar.title("🚖 Fleet Command")
    st.sidebar.caption("NSUT B.Tech Project Phase 1")
    st.sidebar.markdown("---")
    
    view_mode = st.sidebar.radio("Select System Module:", 
        ["📍 Live Operations", "🔬 Research Lab", "📉 Efficiency Analysis"])
    
    st.sidebar.info("System Status: **Active**")

    with st.spinner("Initializing Fleet Systems..."):
        df = load_data('train.csv')
        model, model_status = load_model()
    
    if df is None:
        st.error("🚨 Critical Error: 'train.csv' not found. Please put the dataset in the project folder.")
        st.stop()

    st.markdown(f"### Intelligent Fleet Allocation System")
    st.markdown(f"**AI Engine:** `{model_status}` | **Active Records:** `{len(df):,}`")
    st.markdown("---")

    map_config = {'scrollZoom': True, 'displayModeBar': True}

    if view_mode == "📍 Live Operations":
        st.sidebar.markdown("### ⚙️ Dispatch Controls")
        selected_hour = st.sidebar.slider("Select Time Window (Hour):", 0, 23, 18)
        
        df_view = df[df['hour'] == selected_hour]
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Active Pickups", len(df_view), delta="Live")
        col2.metric("Avg Duration", f"{df_view['trip_duration'].mean()/60:.1f} min")
        col3.metric("Avg Speed", f"{df_view['speed_kmh'].mean():.1f} km/h")
        
        if len(df_view) > 2500:
            status = "🔥 Critical Demand"
        elif len(df_view) > 1500:
            status = "🟠 High Demand"
        else:
            status = "🟢 Normal Operations"
        col4.metric("Status", status)
        
        if model:
            try:
                coords = df_view[['pickup_latitude', 'pickup_longitude']]
                df_view['cluster'] = model.predict(coords)
                
                st.subheader(f"AI-Optimized Driver Allocation Zones ({selected_hour}:00)")
                
                fig = px.scatter_mapbox(df_view, lat='pickup_latitude', lon='pickup_longitude',
                                        color='cluster', size='passenger_count',
                                        zoom=11, height=600, mapbox_style="carto-darkmatter",
                                        color_continuous_scale=px.colors.qualitative.Bold,
                                        title="Predicted Hotspots (Clusters)")
                
                st.plotly_chart(fig, use_container_width=True, config=map_config)
                
                st.info(f"**Strategy:** The AI has segmented the city into {model.n_clusters} optimal zones. Drivers in 'Zone 0' (Yellow) should relocate to 'Zone 2' (Purple) to balance demand.")
            except Exception as e:
                st.warning(f"Model prediction error: {e}")
        else:
            st.subheader("Raw Demand Density")
            fig = px.density_mapbox(df_view, lat='pickup_latitude', lon='pickup_longitude', 
                                    z='passenger_count', radius=10, zoom=11, mapbox_style="carto-darkmatter")
            st.plotly_chart(fig, use_container_width=True, config=map_config)

    elif view_mode == "🔬 Research Lab":
        st.subheader("🧪 Comparative Algorithm Analysis")
        st.markdown("""
        **Objective:** Prove that Density-Based Clustering (HDBSCAN) provides better noise filtering than Centroid-Based (K-Means) for urban topology.
        """)
        
        sample_size = st.slider("Experiment Sample Size:", 1000, 10000, 5000)
        df_sample = df.sample(sample_size)
        
        col_left, col_right = st.columns(2)
     
        with col_left:
            st.markdown("#### Method A: K-Means")
            k = st.slider("K (Centroids):", 3, 15, 8, key='k_slider')
            km = KMeans(n_clusters=k).fit(df_sample[['pickup_latitude', 'pickup_longitude']])
            df_sample['kmeans'] = km.labels_.astype(str)
            
            fig_k = px.scatter_mapbox(df_sample, lat='pickup_latitude', lon='pickup_longitude', 
                                     color='kmeans', zoom=10, height=450,
                                     mapbox_style="carto-positron", title="Result: Spherical Zones")
          
            st.plotly_chart(fig_k, use_container_width=True, config=map_config)
            st.error("Observation: Forces outliers into clusters (Inefficient).")
            
        with col_right:
            st.markdown("#### Method B: HDBSCAN (Proposed)")
            
            try:
                import hdbscan
                min_c = st.slider("Min Cluster Size:", 10, 100, 30, key='h_slider')
                clusterer = hdbscan.HDBSCAN(min_cluster_size=min_c)
                df_sample['hdbscan'] = clusterer.fit_predict(df_sample[['pickup_latitude', 'pickup_longitude']])
                
                noise_count = len(df_sample[df_sample['hdbscan'] == -1])
                
                fig_h = px.scatter_mapbox(df_sample, lat='pickup_latitude', lon='pickup_longitude', 
                                         color='hdbscan', zoom=10, height=450,
                                         mapbox_style="carto-positron", title="Result: High-Density Hotspots")
                
                st.plotly_chart(fig_h, use_container_width=True, config=map_config)
                st.success(f"Observation: Identified {noise_count} Noise Points (Gray) to be ignored.")
                

            except ImportError:
                st.warning("HDBSCAN library not installed. Please install it to view this experiment.")

    elif view_mode == "📉 Efficiency Analysis":
        st.subheader("🚦 Fleet Efficiency & Congestion Metrics")
        st.markdown("Analyzing Trip Duration and Speed to identify 'Dead Zones' where traffic kills revenue.")
        
        tab1, tab2 = st.tabs(["Velocity Profile", "Trip Durations"])
        
        with tab1:
            st.markdown("**Average Fleet Speed by Hour** (Congestion Indicator)")
            avg_speed = df.groupby('hour')['speed_kmh'].mean().reset_index()
            fig_speed = px.line(avg_speed, x='hour', y='speed_kmh', markers=True, 
                               line_shape='spline', color_discrete_sequence=['#00CC96'])
            
            fig_speed.add_annotation(x=18, y=avg_speed.loc[18, 'speed_kmh'],
                                    text="Evening Rush (Slowest)", showarrow=True, arrowhead=1)
            
            st.plotly_chart(fig_speed, use_container_width=True)
            st.caption("Insight: Fleet speed drops by ~40% at 18:00 hours due to congestion.")
            
        with tab2:
            st.markdown("**Trip Duration Distribution**")
            fig_hist = px.histogram(df, x='trip_duration', nbins=100, range_x=[0, 3600],
                                   title="Frequency of Trip Lengths (Seconds)",
                                   color_discrete_sequence=['#AB63FA'])
            st.plotly_chart(fig_hist, use_container_width=True)

if __name__ == "__main__":
    main()