import streamlit as st
import pandas as pd
from scipy.signal import find_peaks
from prophet import Prophet
import plotly.graph_objects as go

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(page_title="🌸 BloomWatch", layout="wide", page_icon="🌸")

# ---------------------------
# App Title
# ---------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#FF69B4;'>🌸 BloomWatch – Multi-Region NDVI Prototype</h1>
    <p style='text-align:center; font-size:18px;'>Track and predict blooming events 🌱✨</p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ---------------------------
# Step 1: Region Selection
# ---------------------------
st.subheader("📍 Step 1 – Select a Region")
regions = {
    "Bhopal": "sample_ndvi_bhopal.csv",
    "Indore": "sample_ndvi_indore.csv",
    "Jabalpur": "sample_ndvi_jabalpur.csv",
    "Gwalior": "sample_ndvi_gwalior.csv",
    "Ujjain": "sample_ndvi_ujjain.csv",
    "Sagar": "sample_ndvi_sagar.csv",
    "Rewa": "sample_ndvi_rewa.csv",
    "Satna": "sample_ndvi_satna.csv"
}

col1, col2 = st.columns([2, 1])
with col1:
    region = st.selectbox("Choose a region:", list(regions.keys()))
with col2:
    st.info("🌎 Select a region to load its NDVI dataset.")

@st.cache_data
def load_data(filename):
    df = pd.read_csv(filename)
    df['date'] = pd.to_datetime(df['date'])
    return df.sort_values('date').reset_index(drop=True)

df = load_data(regions[region])
st.success(f"✅ NDVI dataset for **{region}** loaded successfully.")

st.markdown("---")

# ---------------------------
# Step 2: Detect Past Blooms
# ---------------------------
with st.expander("🌸 Step 2 – Detecting Past Bloom Events", expanded=True):
    df['ndvi_smooth'] = df['ndvi'].rolling(3, min_periods=1, center=True).mean()
    vals = df['ndvi_smooth'].values
    peaks, _ = find_peaks(vals, height=0.6, distance=8)
    df['is_peak'] = False
    df.loc[peaks, 'is_peak'] = True

    if len(peaks) > 0:
        st.success(f"🌸 Found {len(peaks)} past bloom events in {region}:")
        bloom_dates = df.loc[peaks, 'date'].dt.strftime('%d/%m/%Y').tolist()
        st.markdown(
            " | ".join([f"**{d}**" for d in bloom_dates]),
            unsafe_allow_html=True
        )
    else:
        st.warning("⚠️ No bloom events detected in historical data.")

# ---------------------------
# Step 3: Predict Future Blooms
# ---------------------------
with st.expander("🔮 Step 3 – Predicting Future Bloom Events", expanded=True):
    prophet_df = df[['date','ndvi_smooth']].rename(columns={'date':'ds','ndvi_smooth':'y'}).dropna()
    m = Prophet(yearly_seasonality=True)
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=12, freq='16D')
    forecast = m.predict(future)

    forecast_vals = forecast[['ds','yhat']].copy()
    forecast_vals['yhat_smooth'] = forecast_vals['yhat'].rolling(3, min_periods=1, center=True).mean()
    f_peaks, _ = find_peaks(forecast_vals['yhat_smooth'].values, height=0.6, distance=8)
    future_peaks = forecast_vals.iloc[f_peaks]
    future_peaks = future_peaks[future_peaks['ds'] > df['date'].max()]

    if not future_peaks.empty:
        st.success("🔮 Predicted future bloom events:")
        for i, row in future_peaks.head(3).iterrows():
            st.markdown(
                f"⏳ **Expected:** {row['ds'].strftime('%d/%m/%Y')}  |  🌱 **NDVI:** {row['yhat']:.3f}"
            )
    else:
        st.warning("⚠️ No future bloom predicted.")

# ---------------------------
# Step 4: Visualization
# ---------------------------
with st.expander("📊 Step 4 – Bloom Timeline Visualization", expanded=True):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['ndvi'], mode='lines+markers',
                             name='NDVI', line=dict(color='lightgreen')))
    fig.add_trace(go.Scatter(x=df['date'], y=df['ndvi_smooth'], mode='lines',
                             name='Smoothed NDVI', line=dict(color='green')))
    if len(peaks)>0:
        fig.add_trace(go.Scatter(x=df.loc[peaks,'date'], y=df.loc[peaks,'ndvi_smooth'],
                                 mode='markers', marker=dict(color='red', size=10),
                                 name='Detected Blooms'))
    fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'],
                             mode='lines', name='Forecast NDVI',
                             line=dict(color='blue', dash='dash')))
    if not future_peaks.empty:
        fig.add_trace(go.Scatter(x=future_peaks['ds'], y=future_peaks['yhat'],
                                 mode='markers', marker=dict(color='orange', size=9),
                                 name='Predicted Blooms'))
    fig.update_layout(height=450, xaxis_title="Date", yaxis_title="NDVI",
                      plot_bgcolor='black')
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------
# Step 5: Map
# ---------------------------
with st.expander("🗺️ Step 5 – Region Map", expanded=False):
    if set(['lat','lon']).issubset(df.columns):
        st.map(df[['lat','lon']].dropna())
    else:
        st.info("📍 No lat/lon data available for this region.")

# ---------------------------
# Step 6: Export Data
# ---------------------------
with st.expander("⬇️ Step 6 – Export Results", expanded=False):
    out = df[['date','ndvi','ndvi_smooth','is_peak']].copy()
    out['date'] = out['date'].dt.strftime('%d/%m/%Y')
    csv = out.to_csv(index=False)
    st.download_button("💾 Download Bloom Detection CSV", csv,
                       file_name=f"bloom_peaks_{region.lower()}.csv", mime="text/csv")
