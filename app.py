import streamlit as st
import pandas as pd
from scipy.signal import find_peaks
from prophet import Prophet
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------
# App Config
# ---------------------------
st.set_page_config(page_title="🌸 BloomWatch", layout="wide", page_icon="🌸")

# ---------------------------
# Data and Constants
# ---------------------------
REGIONS = {
    "Bhopal": "sample_ndvi_bhopal.csv",
    "Indore": "sample_ndvi_indore.csv",
    "Jabalpur": "sample_ndvi_jabalpur.csv",
    "Gwalior": "sample_ndvi_gwalior.csv",
    "Ujjain": "sample_ndvi_ujjain.csv",
    "Sagar": "sample_ndvi_sagar.csv",
    "Rewa": "sample_ndvi_rewa.csv",
    "Satna": "sample_ndvi_satna.csv"
}

# ---------------------------
# Helper Functions
# ---------------------------
@st.cache_data
def load_data(filename):
    """Loads and preprocesses NDVI data."""
    try:
        df = pd.read_csv(filename)
        df['date'] = pd.to_datetime(df['date'])
        return df.sort_values('date').reset_index(drop=True)
    except FileNotFoundError:
        st.error(f"Error: The file '{filename}' was not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the data: {e}")
        st.stop()

def detect_blooms(df):
    """Detects past bloom events using a smoothed NDVI curve."""
    df['ndvi_smooth'] = df['ndvi'].rolling(window=3, min_periods=1, center=True).mean()
    peaks, _ = find_peaks(df['ndvi_smooth'].values, height=0.6, distance=8)
    df['is_peak'] = False
    df.loc[peaks, 'is_peak'] = True
    return df, peaks

def predict_future_blooms(df):
    """Predicts future NDVI trends and potential bloom events."""
    prophet_df = df[['date', 'ndvi_smooth']].rename(columns={'date': 'ds', 'ndvi_smooth': 'y'}).dropna()
    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=12, freq='16D')
    forecast = m.predict(future)
    
    forecast_vals = forecast[['ds', 'yhat']].copy()
    forecast_vals['yhat_smooth'] = forecast_vals['yhat'].rolling(window=3, min_periods=1, center=True).mean()
    future_peaks_indices, _ = find_peaks(forecast_vals['yhat_smooth'].values, height=0.6, distance=8)
    future_peaks = forecast_vals.iloc[future_peaks_indices]
    
    # Filter for future dates
    future_peaks = future_peaks[future_peaks['ds'] > df['date'].max()]
    return forecast, future_peaks

def create_plot(df, peaks, forecast, future_peaks):
    """Generates the main Plotly visualization."""
    fig = go.Figure()
    
    # Plot historical NDVI and smoothed curve
    fig.add_trace(go.Scatter(x=df['date'], y=df['ndvi'], mode='lines', name='Raw NDVI', line=dict(color='lightgreen', width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=df['date'], y=df['ndvi_smooth'], mode='lines', name='Smoothed NDVI', line=dict(color='green', width=2)))

    # Plot detected historical blooms
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(x=df.loc[peaks, 'date'], y=df.loc[peaks, 'ndvi_smooth'], mode='markers', name='Detected Blooms', marker=dict(color='red', size=10, symbol='star', line=dict(color='white', width=1.5))))
    
    # Plot predicted NDVI forecast
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast', line=dict(color='blue', dash='dash', width=2)))
    
    # Plot predicted future blooms
    if not future_peaks.empty:
        fig.add_trace(go.Scatter(x=future_peaks['ds'], y=future_peaks['yhat'], mode='markers', name='Predicted Blooms', marker=dict(color='orange', size=10, symbol='circle', line=dict(color='white', width=1.5))))

    # Update layout for a cleaner look
    fig.update_layout(
        title=f"NDVI Timeline for {st.session_state.get('region', 'Selected Region')}",
        xaxis_title="Date",
        yaxis_title="NDVI",
        template='plotly_dark',
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🌸 BloomWatch – Multi-Region NDVI Prototype")
st.markdown("Track and predict blooming events using satellite-based NDVI data. 🌱✨")
st.markdown("---")

# Use st.sidebar for region selection for better UI
with st.sidebar:
    st.header("Settings")
    region = st.selectbox("📍 Select a Region", list(REGIONS.keys()))
    st.info("🌎 Choose a region to analyze its Normalized Difference Vegetation Index (NDVI) data.")
    st.markdown("---")
    
# Store the selected region in session state to maintain consistency
if 'region' not in st.session_state or st.session_state.region != region:
    st.session_state.region = region
    
# Main app content
st.header("Step 1: Data Loading")
df = load_data(REGIONS[st.session_state.region])
st.success(f"✅ NDVI dataset for **{st.session_state.region}** loaded successfully.")

# ---
st.header("Step 2: Detecting Past Blooms")
st.info("We use a peak detection algorithm to identify past bloom events in the historical data.")
df, peaks = detect_blooms(df)
if len(peaks) > 0:
    st.success(f"🌸 Found {len(peaks)} past bloom events in {st.session_state.region}:")
    bloom_dates = df.loc[peaks, 'date'].dt.strftime('%d/%m/%Y').tolist()
    st.markdown(
        " | ".join([f"**{d}**" for d in bloom_dates]),
        unsafe_allow_html=True
    )
else:
    st.warning("⚠️ No bloom events detected in historical data.")

# ---
st.header("Step 3: Predicting Future Blooms")
st.info("Using Facebook's Prophet library, we forecast the NDVI trend and predict future bloom dates.")
with st.spinner("Forecasting future blooms..."):
    forecast, future_peaks = predict_future_blooms(df)
    
if not future_peaks.empty:
    st.success("🔮 Predicted future bloom events:")
    for i, row in future_peaks.head(3).iterrows():
        st.markdown(
            f"⏳ **Expected:** {row['ds'].strftime('%d/%m/%Y')} &nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp; 🌱 **NDVI:** {row['yhat']:.3f}"
        )
else:
    st.warning("⚠️ No future bloom predicted.")

# ---
st.header("Step 4: Visualization")
st.info("Explore the complete timeline, including historical data, detected blooms, and future predictions.")
fig = create_plot(df, peaks, forecast, future_peaks)
st.plotly_chart(fig, use_container_width=True)

# ---
st.header("Step 5: Export Results")
st.info("Download the processed historical data with detected bloom events.")
out = df[['date', 'ndvi', 'ndvi_smooth', 'is_peak']].copy()
out['date'] = out['date'].dt.strftime('%d/%m/%Y')
csv = out.to_csv(index=False)
st.download_button(
    "💾 Download Processed Data",
    csv,
    file_name=f"bloom_data_{st.session_state.region.lower()}.csv",
    mime="text/csv"
)

# ---
st.subheader("Map of the Region")
if set(['lat', 'lon']).issubset(df.columns):
    st.map(df[['lat', 'lon']].dropna())
else:
    st.info("📍 Latitude and longitude data not available for this region's map.")
