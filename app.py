import os
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

APP_TITLE = "Talabat Delivery Intelligence Dashboard"

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🛵",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================

st.markdown("""
<style>

:root {
    --talabat-orange:#ff5a00;
    --dark:#111111;
    --soft:#fff7f2;
    --muted:#666;
}

.stApp {
    background:
    linear-gradient(
        135deg,
        #fff 0%,
        #fff7f2 45%,
        #ffffff 100%
    );
}

.block-container {
    padding-top: 1.4rem;
    padding-bottom: 2rem;
}

[data-testid="stSidebar"] {
    background: #111111;
}

[data-testid="stSidebar"] * {
    color: white !important;
}

.hero {
    padding: 28px;
    border-radius: 28px;
    color: white;

    background:
    radial-gradient(
        circle at top right,
        #ff944d,
        #ff5a00 45%,
        #111 115%
    );

    box-shadow:
    0 18px 45px rgba(255,90,0,.20);

    margin-bottom: 20px;
}

.hero h1 {
    font-size: 42px;
    margin-bottom: 8px;
}

.hero p {
    font-size: 17px;
    opacity:.95;
    max-width: 980px;
}

.card {
    background: rgba(255,255,255,.88);

    border:
    1px solid rgba(255,90,0,.12);

    border-radius: 24px;

    padding: 20px;

    box-shadow:
    0 12px 32px rgba(17,17,17,.07);
}

.small-card {
    background: white;

    border-left:
    6px solid #ff5a00;

    border-radius: 20px;

    padding: 18px 20px;

    box-shadow:
    0 8px 24px rgba(17,17,17,.06);
}

.badge {
    display:inline-block;
    padding:7px 12px;
    border-radius:999px;
    background:#111;
    color:#fff;
    font-weight:700;
}

.warning-box {
    background:#fff3e8;
    border:1px solid #ffbf91;
    border-radius:18px;
    padding:16px;
}

.success-box {
    background:#effaf3;
    border:1px solid #afe0bd;
    border-radius:18px;
    padding:16px;
}

.metric-label {
    color:#666;
    font-size:14px;
}

.metric-value {
    color:#111;
    font-size:28px;
    font-weight:800;
}

hr {
    margin: 1.2rem 0;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# FEATURES
# =====================================================

CUSTOMER_FEATURES = [
    "Total_Price_std",
    "Quantity_std",
    "Order_Hour_std",
    "Payment_Method_Encoded",
    "Traffic_Level_Encoded",
    "Total_Delivery_Route"
]

PERFORMANCE_FEATURES = [
    "Delivery_Duration_Minutes_std",
    "Total_Delivery_Route",
    "Traffic_Level_Encoded",
    "Driver_Vehicle_Encoded",
    "Delivery_Hour_std"
]

PAYMENT_MAP = {
    "Cash": 0,
    "Card": 1,
    "Credit Card": 1,
    "Wallet": 2
}

TRAFFIC_MAP = {
    "Low": 0,
    "Medium": 1,
    "High": 2
}

VEHICLE_MAP = {
    "Bicycle": 0,
    "Bike": 1,
    "Motorbike": 1,
    "Car": 2
}

# =====================================================
# EVALUATION TABLES
# =====================================================

CUSTOMER_EVAL = pd.DataFrame({
    "Model": ["K-Means", "GMM", "DBSCAN"],
    "Silhouette Score": [0.481242, 0.480522, 0.299869],
    "Davies-Bouldin Index": [0.695353, np.nan, np.nan],
    "Calinski-Harabasz Score": [106038.562645, np.nan, np.nan],
    "Use Case": ["Main model", "Comparison", "Comparison"]
})

PERF_EVAL = pd.DataFrame({
    "Model": ["K-Means", "GMM", "DBSCAN"],
    "Silhouette Score": [0.440408, 0.440370, 0.388977],
    "Davies-Bouldin Index": [0.782154, np.nan, np.nan],
    "Calinski-Harabasz Score": [102654.998004, np.nan, np.nan],
    "Use Case": ["Main model", "Comparison", "Comparison"]
})

# =====================================================
# CLUSTER INFO
# =====================================================

CUSTOMER_CLUSTER_INFO = {
    0: {
        "name":"Value Regular Customer",
        "desc":"Pelanggan dengan transaksi stabil.",
        "rec":"Berikan loyalty points dan promo ringan."
    },

    1: {
        "name":"High Value Customer",
        "desc":"Pelanggan bernilai tinggi.",
        "rec":"Gunakan premium promo dan subscription."
    },

    2: {
        "name":"Quick Small Order Customer",
        "desc":"Pelanggan dengan order cepat dan kecil.",
        "rec":"Tawarkan quick reorder dan promo ongkir."
    },

    3: {
        "name":"Route-Sensitive Customer",
        "desc":"Pelanggan sensitif terhadap route dan traffic.",
        "rec":"Optimalkan estimasi waktu dan driver."
    },
}

PERF_CLUSTER_INFO = {
    0: {
        "name":"Efficient Delivery",
        "desc":"Durasi dan route efisien.",
        "insight":"Pertahankan strategi saat ini."
    },

    1: {
        "name":"Moderate Delivery",
        "desc":"Performa normal.",
        "insight":"Masih dapat dioptimasi."
    },

    2: {
        "name":"Traffic Heavy Delivery",
        "desc":"Terdampak traffic atau route.",
        "insight":"Perlu route optimization."
    },

    3: {
        "name":"Slow Delivery Risk",
        "desc":"Berisiko lambat.",
        "insight":"Perlu perhatian khusus."
    },
}

# =====================================================
# LOAD MODELS
# =====================================================

@st.cache_resource
def load_models():

    required = {
        "customer_kmeans": "customer_kmeans.pkl",
        "customer_scaler": "customer_scaler.pkl",
        "customer_pca": "customer_pca.pkl",

        "performance_kmeans": "performance_kmeans.pkl",
        "performance_scaler": "performance_scaler.pkl",
        "performance_pca": "performance_pca.pkl",
    }

    missing = [
        name
        for name in required.values()
        if not (MODEL_DIR / name).exists()
    ]

    if missing:
        st.error(
            "File model belum lengkap.\n"
            "Jalankan python model_training.py terlebih dahulu."
        )
        st.stop()

    return {
        key: joblib.load(MODEL_DIR / filename)
        for key, filename in required.items()
    }

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_sample_data():

    path = DATA_DIR / "sample_talabat.csv"

    if path.exists():

        df = pd.read_csv(path)

        # =========================================
        # FIX DATE PARSING WARNING
        # =========================================

        if "Order_Time" in df.columns:
            df["Order_Time"] = pd.to_datetime(
                df["Order_Time"],
                format="%d/%m/%Y %H:%M",
                errors="coerce"
            )

        if "Delivery_Time" in df.columns:
            df["Delivery_Time"] = pd.to_datetime(
                df["Delivery_Time"],
                format="%d/%m/%Y %H:%M",
                errors="coerce"
            )

        return df

    # =========================================
    # FALLBACK RANDOM DATA
    # =========================================

    rng = np.random.default_rng(42)

    n = 500

    df = pd.DataFrame({
        "Total_Price":
            rng.gamma(3, 25, n).round(2),

        "Quantity":
            rng.integers(1, 8, n),

        "Order_Hour":
            rng.integers(0, 24, n),

        "Delivery_Hour":
            rng.integers(0, 24, n),

        "Delivery_Duration_Minutes":
            np.clip(
                rng.normal(38, 10, n),
                10,
                90
            ).round(1),

        "Total_Delivery_Route":
            np.clip(
                rng.normal(8, 3, n),
                1,
                22
            ).round(2),

        "Traffic_Level":
            rng.choice(
                ["Low", "Medium", "High"],
                n,
                p=[.35,.45,.20]
            ),

        "Payment_Method":
            rng.choice(
                ["Cash", "Card", "Wallet"],
                n
            ),

        "Driver_Vehicle":
            rng.choice(
                ["Bicycle", "Bike", "Car"],
                n,
                p=[.20,.60,.20]
            ),
    })

    return df

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def hero(title, subtitle):

    st.markdown(
        f"""
        <div class='hero'>
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def metric_card(label, value):

    st.markdown(
        f"""
        <div class='small-card'>
            <div class='metric-label'>{label}</div>
            <div class='metric-value'>{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# =====================================================
# LOAD APP
# =====================================================

models = load_models()
df = load_sample_data()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.markdown("# 🧡 Talabat ML")

page = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Customer Segmentation",
        "Delivery Performance",
        "Model Evaluation",
        "Data Insight",
        "About"
    ]
)

# =====================================================
# HOME
# =====================================================

if page == "Home":

    hero(
        "🛵 Talabat Delivery Intelligence Dashboard",
        "Customer segmentation & delivery clustering dashboard."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card("Total Data", f"{len(df):,}")

    with c2:
        metric_card("Features", "11")

    with c3:
        metric_card("Best Model", "K-Means")

    with c4:
        metric_card("Best Silhouette", "0.481")

    st.markdown("""
    <div class='card'>
    Dashboard machine learning interaktif menggunakan:
    <ul>
        <li>K-Means</li>
        <li>GMM</li>
        <li>DBSCAN</li>
        <li>PCA</li>
        <li>Streamlit</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# =====================================================
# CUSTOMER PAGE
# =====================================================

elif page == "Customer Segmentation":

    hero(
        "👥 Customer Segmentation",
        "Prediksi cluster customer."
    )

    st.info("Customer segmentation page berhasil load.")

# =====================================================
# PERFORMANCE PAGE
# =====================================================

elif page == "Delivery Performance":

    hero(
        "⚡ Delivery Performance",
        "Analisis performa delivery."
    )

    st.info("Performance clustering page berhasil load.")

# =====================================================
# EVALUATION PAGE
# =====================================================

elif page == "Model Evaluation":

    hero(
        "📊 Model Evaluation",
        "Perbandingan K-Means, GMM, dan DBSCAN."
    )

    st.subheader("Customer Evaluation")
    st.dataframe(CUSTOMER_EVAL, use_container_width=True)

    st.subheader("Performance Evaluation")
    st.dataframe(PERF_EVAL, use_container_width=True)

# =====================================================
# DATA INSIGHT
# =====================================================

elif page == "Data Insight":

    hero(
        "🔎 Data Insight",
        "Preview dataset dan insight."
    )

    st.dataframe(df.head(20), use_container_width=True)

    fig = px.histogram(
        df,
        x="Total_Price",
        nbins=35,
        title="Distribution of Total Price",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

# =====================================================
# ABOUT
# =====================================================

else:

    hero(
        "ℹ️ About",
        "Tentang aplikasi."
    )

    st.markdown("""
    <div class='card'>
    Dashboard dibuat menggunakan:
    <ul>
        <li>Streamlit</li>
        <li>Scikit-Learn</li>
        <li>Plotly</li>
        <li>PCA</li>
        <li>K-Means</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
