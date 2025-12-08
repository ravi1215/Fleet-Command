# 🚖 Intelligent Fleet Allocation System (Phase 1)

> **B.Tech Major Project | Department of Computer Science & Engineering | NSUT*

This repository contains the source code and research methodologies for the **Intelligent Fleet Allocation System**, a machine learning-based solution designed to optimize urban taxi fleet positioning.

The system utilizes **Unsupervised Learning** (Clustering) to identify high-demand "hotspots" in New York City, enabling predictive driver dispatch and reducing idle time.

---

## 🚀 Features (Phase 1)

### 1. 📍 Live Operations Dashboard
* **Real-time Hotspot Prediction:** Uses a pre-trained **K-Means model** to predict optimal waiting zones for drivers.
* **Dynamic filtering:** Analyze demand by hour of the day (e.g., Morning Rush vs. Evening Commute).
* **Status Alerts:** Automatic detection of "Critical Demand" periods.

### 2. 🔬 Research Laboratory
* **Algorithm Comparison:** A comparative study between **Centroid-based (K-Means)** and **Density-based (HDBSCAN)** clustering.
* **Noise Filtering:** Demonstrates HDBSCAN's ability to filter out 10% of GPS outliers (noise) to prevent inefficient dispatch.

### 3. 📉 Efficiency Metrics
* **Velocity Profiling:** Analyzes fleet speed vs. time to identify congestion bottlenecks.
* **Trip Duration Analysis:** Histogram analysis of trip lengths to optimize short-haul vs. long-haul assignments.

---

## 🛠️ Tech Stack
* **Language:** Python 3.10+
* **Frontend:** Streamlit (Web App)
* **ML Core:** Scikit-Learn, HDBSCAN
* **Visualization:** Plotly Mapbox, Seaborn
* **Data Processing:** Pandas, NumPy
* **Model Serialization:** Joblib

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher installed.
* Download the **NYC Taxi Trip Duration** dataset (only `train.csv` is required) from [Kaggle](https://www.kaggle.com/c/nyc-taxi-trip-duration/data).

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/fleet-allocation-system.git](https://github.com/your-username/fleet-allocation-system.git)
cd fleet-allocation-system
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```
### 3. Add the Dataset
Due to GitHub file size limits, the dataset is not included.
Place your train.csv file inside the root project folder.

### 4. Run the Application
```bash
streamlit run app.py
```

