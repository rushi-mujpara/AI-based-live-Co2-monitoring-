import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
import joblib
import json

# --- 1. CONFIGURATION & ASSETS ---
API_KEY = "icZAcI0AUk00RmRF2L4O59PBfPphPWIv"
SURAT_CENTER = [21.1702, 72.8311]

st.set_page_config(page_title="Eco-Traffic Predictor", layout="wide")

@st.cache_resource
def load_ml_assets():
    model = joblib.load('traffic_model.pkl')
    with open('model_metadata.json', 'r') as f:
        metadata = json.load(f)
    return model, metadata

try:
    model, metadata = load_ml_assets()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# --- 2. LOGIC FUNCTIONS ---
def get_segment_data(lat, lon):
    url = f"https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json?point={lat},{lon}&key={API_KEY}"
    response = requests.get(url)
    if response.status_status == 200:
        return response.json().get('flowSegmentData')
    return None

def predict_co2(speed, free_flow_speed):
    # Bridge: Estimate consumption based on congestion ratio
    # Baseline for a 1.5L car is ~8.0 L/100km
    congestion_ratio = speed / free_flow_speed if free_flow_speed > 0 else 1
    
    if congestion_ratio < 0.3:
        consumption = 15.0  # Heavy traffic idling
    elif congestion_ratio < 0.7:
        consumption = 10.5  # Moderate traffic
    else:
        consumption = 8.0   # Optimal flow
        
    # Create input matching the training features (XGBoost expects these columns)
    input_data = {
        'Engine Size(L)': 1.5,
        'Cylinders': 4,
        'Fuel Consumption Comb (L/100 km)': consumption,
        'Fuel Type_X': 1 # Assuming Regular Gasoline
    }
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=metadata['features'], fill_value=0)
    return model.predict(input_df)[0]

# --- 3. UI LAYOUT ---
st.title("🍃 Real-Time Traffic Emission Predictor")
st.markdown("Click on any **colored traffic line** on the map to analyze the Carbon Footprint of that road segment.")

col1, col2 = st.columns([3, 1])

with col1:
    # Initialize Folium Map
    m = folium.Map(location=SURAT_CENTER, zoom_start=13, tiles="cartodbpositron")

    # Add TomTom Raster Traffic Tiles
    traffic_url = (
        f"https://api.tomtom.com/traffic/map/4/tile/flow/absolute/{{z}}/{{x}}/{{y}}.png"
        f"?key={API_KEY}&thickness=10"
    )
    
    folium.TileLayer(
        tiles=traffic_url,
        attr='TomTom Traffic',
        name='Live Traffic Flow',
        overlay=True,
        opacity=0.8
    ).add_to(m)

    # Render Map and catch clicks
    map_data = st_folium(m, width="100%", height=600)

with col2:
    st.subheader("📊 Segment Analytics")
    
    if map_data and map_data.get('last_clicked'):
        lat = map_data['last_clicked']['lat']
        lon = map_data['last_clicked']['lng']
        
        with st.spinner("Fetching live traffic..."):
            segment = get_segment_data(lat, lon)
            
        if segment:
            speed = segment['currentSpeed']
            ff_speed = segment['freeFlowSpeed']
            co2_val = predict_co2(speed, ff_speed)
            
            st.metric("Live Speed", f"{speed} km/h")
            st.metric("CO2 Emission", f"{round(co2_val, 2)} g/km")
            
            # Impact Gauge logic
            if co2_val > 220:
                st.error("Critical: High Emission Zone")
            elif co2_val > 160:
                st.warning("Moderate: Congested")
            else:
                st.success("Efficient: Low Impact")
                
            st.info(f"Location: {lat:.4f}, {lon:.4f}")
        else:
            st.warning("No traffic data found for this point. Try clicking directly on a road.")
    else:
        st.info("Click a road on the map to begin analysis.")

# --- 4. HISTORICAL INSIGHTS (FOOTER) ---
st.divider()
st.subheader("📈 Environmental Insights")
# This is a placeholder for your logged data
chart_data = pd.DataFrame({
    'Time': ["08:00", "10:00", "12:00", "14:00", "16:00", "18:00", "20:00"],
    'Avg CO2 (g/km)': [210, 180, 155, 160, 190, 240, 175]
})
st.line_chart(chart_data.set_index('Time'))