# 🌸 BloomWatch – Prototype

## 📌 Overview
BloomWatch is a **satellite-data powered prototype** that detects and predicts blooming events in vegetation using the **Normalized Difference Vegetation Index (NDVI)**.  
The system leverages **historical NDVI data** to detect past bloom events 🌱 and uses **time-series forecasting** to predict upcoming bloom cycles 🔮.  

This prototype is designed for the **NASA Space Apps Challenge 2025** and currently demonstrates the solution with **synthetic NDVI datasets for 8 regions in Madhya Pradesh, India**.

---

## 🚀 Features
- 📍 **Multi-region support** – Select from 8 demo regions (Bhopal, Indore, Jabalpur, Gwalior, Ujjain, Sagar, Rewa, Satna).  
- 🌸 **Bloom detection** – Identifies past bloom events using NDVI peak detection.  
- 🔮 **Bloom prediction** – Forecasts future bloom events using **Facebook Prophet**.  
- 📊 **Interactive visualization** – Timeline plots with raw NDVI, smoothed NDVI, detected blooms, and future predictions.  
- 🗺️ **Regional mapping** – Simple map to locate regions.  
- 💾 **Data export** – Download processed bloom data as CSV.  

---

## 🛠️ Tech Stack
- **Frontend/UI:** Streamlit + Plotly  
- **Backend/Processing:** Python, Pandas, SciPy, Prophet  
- **Database (demo):** CSV datasets (synthetic NDVI data)  
- **Mapping:** Streamlit map, Leaflet (future scope)  
- **Hosting:** Streamlit Cloud / Local Run  
- **Scalability:** PostgreSQL + PostGIS, AWS/Azure (future production)  

---

## 🌍 Data Sources
- Prototype currently uses **synthetic NDVI datasets (2023–2024)** for demo.  
- The system is fully **extendable to real satellite datasets**:  
  - 🛰️ **NASA MODIS Terra/Aqua NDVI** (MOD13Q1, MOD09GA)  
  - 🛰️ **ESA Sentinel-2 Surface Reflectance**  
- Integration via **Google Earth Engine (GEE) API** will allow automatic NDVI time-series extraction for any location globally.  

---

## ⚡ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/<your-repo-name>/bloomwatch.git
cd bloomwatch
