import streamlit as st
import pandas as pd
from scipy.signal import find_peaks
from prophet import Prophet
import plotly.graph_objects as go

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
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

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
    fig.add_trace(go.Scatter(x=df['date'], y=df['ndvi'], mode='lines',
                             name='Raw NDVI', line=dict(color='lightgreen', width=1), opacity=0.6))
    fig.add_trace(go.Scatter(x=df['date'], y=df['ndvi_smooth'], mode='lines',
                             name='Smoothed NDVI', line=dict(color='green', width=2)))

    # Plot detected historical blooms
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(x=df.loc[peaks, 'date'], y=df.loc[peaks, 'ndvi_smooth'],
                                 mode='markers', name='Detected Blooms',
                                 marker=dict(color='red', size=10, symbol='star',
                                             line=dict(color='white', width=1.5))))

    # Plot predicted NDVI forecast
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines',
                             name='Forecast', line=dict(color='blue', dash='dash', width=2)))

    # Plot predicted future blooms
    if not future_peaks.empty:
        fig.add_trace(go.Scatter(x=future_peaks['ds'], y=future_peaks['yhat'],
                                 mode='markers', name='Predicted Blooms',
                                 marker=dict(color='orange', size=10, symbol='circle',
                                             line=dict(color='white', width=1.5))))

    # Update layout
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
st.title("🌸 BloomWatch – Multi-Region NDVI")
st.markdown("Track and predict blooming events using satellite-based NDVI data. 🌱✨")
st.markdown("---")

# Region selection (main page)
region = st.selectbox("📍 Select a Region", list(REGIONS.keys()))
if 'region' not in st.session_state or st.session_state.region != region:
    st.session_state.region = region

# Data Loading
df = load_data(REGIONS[st.session_state.region])
st.success(f"✅ NDVI dataset for **{st.session_state.region}** loaded successfully.")

# Detecting Past Blooms
st.subheader("Past Bloom Events")
df, peaks = detect_blooms(df)
if len(peaks) > 0:
    st.success(f"🌸 Found {len(peaks)} past bloom events in {st.session_state.region}")
    bloom_dates = df.loc[peaks, ['date', 'ndvi_smooth']]
    bloom_dates['date'] = bloom_dates['date'].dt.strftime('%d/%m/%Y')
    bloom_dates.rename(columns={'date': 'Bloom Date', 'ndvi_smooth': 'NDVI Value'}, inplace=True)
    st.table(bloom_dates.reset_index(drop=True))
else:
    st.warning("⚠️ No bloom events detected in historical data.")

# Predicting Future Blooms
st.subheader("Predicted Future Blooms")
with st.spinner("Forecasting future blooms..."):
    forecast, future_peaks = predict_future_blooms(df)

if not future_peaks.empty:
    st.success("🔮 Predicted future bloom events:")
    future_table = future_peaks[['ds', 'yhat']].copy()
    future_table['ds'] = future_table['ds'].dt.strftime('%d/%m/%Y')
    future_table.rename(columns={'ds': 'Predicted Date', 'yhat': 'Forecasted NDVI'}, inplace=True)
    st.table(future_table.reset_index(drop=True).head(5))
else:
    st.warning("⚠️ No future bloom predicted.")

# Visualization
st.subheader("NDVI Timeline")
fig = create_plot(df, peaks, forecast, future_peaks)
st.plotly_chart(fig, use_container_width=True)

# Export Results
st.subheader("Export Results")
out = df[['date', 'ndvi', 'ndvi_smooth', 'is_peak']].copy()
out['date'] = out['date'].dt.strftime('%d/%m/%Y')
csv = out.to_csv(index=False)
st.download_button(
    "💾 Download Processed Data",
    csv,
    file_name=f"bloom_data_{st.session_state.region.lower()}.csv",
    mime="text/csv"
)

# Map
st.subheader("Map of the Region")
if set(['lat', 'lon']).issubset(df.columns):
    st.map(df[['lat', 'lon']].dropna())
else:
    st.info("📍 Latitude and longitude data not available for this region's map.")

# Footer
st.markdown("---")
st.caption("🌸 Prototype uses synthetic NDVI data for demo. Extendable to real NASA MODIS / Sentinel-2 datasets via Google Earth Engine API.")
