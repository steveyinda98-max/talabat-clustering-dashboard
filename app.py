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

st.set_page_config(page_title=APP_TITLE, page_icon="🛵", layout="wide")

# =========================
# Custom CSS - Talabat theme
# =========================
st.markdown("""
<style>
:root { --talabat-orange:#ff5a00; --dark:#111111; --soft:#fff7f2; --muted:#666; }
.stApp { background: linear-gradient(135deg,#fff 0%,#fff7f2 45%,#ffffff 100%); }
.block-container { padding-top: 1.4rem; padding-bottom: 2rem; }
[data-testid="stSidebar"] { background: #111111; }
[data-testid="stSidebar"] * { color: white !important; }
.hero {
    padding: 28px; border-radius: 28px; color: white;
    background: radial-gradient(circle at top right, #ff944d, #ff5a00 45%, #111 115%);
    box-shadow: 0 18px 45px rgba(255,90,0,.20); margin-bottom: 20px;
}
.hero h1 { font-size: 42px; margin-bottom: 8px; }
.hero p { font-size: 17px; opacity:.95; max-width: 980px; }
.card {
    background: rgba(255,255,255,.88); border: 1px solid rgba(255,90,0,.12);
    border-radius: 24px; padding: 20px; box-shadow: 0 12px 32px rgba(17,17,17,.07);
}
.small-card {
    background: white; border-left: 6px solid #ff5a00; border-radius: 20px;
    padding: 18px 20px; box-shadow: 0 8px 24px rgba(17,17,17,.06);
}
.badge { display:inline-block; padding:7px 12px; border-radius:999px; background:#111; color:#fff; font-weight:700; }
.warning-box { background:#fff3e8; border:1px solid #ffbf91; border-radius:18px; padding:16px; }
.success-box { background:#effaf3; border:1px solid #afe0bd; border-radius:18px; padding:16px; }
.metric-label { color:#666; font-size:14px; }
.metric-value { color:#111; font-size:28px; font-weight:800; }
hr { margin: 1.2rem 0; }
</style>
""", unsafe_allow_html=True)

CUSTOMER_FEATURES = [
    "Total_Price_std", "Quantity_std", "Order_Hour_std",
    "Payment_Method_Encoded", "Traffic_Level_Encoded", "Total_Delivery_Route"
]
PERFORMANCE_FEATURES = [
    "Delivery_Duration_Minutes_std", "Total_Delivery_Route",
    "Traffic_Level_Encoded", "Driver_Vehicle_Encoded", "Delivery_Hour_std"
]

PAYMENT_MAP = {"Cash": 0, "Card": 1, "Credit Card": 1, "Wallet": 2}
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2}
VEHICLE_MAP = {"Bicycle": 0, "Bike": 1, "Motorbike": 1, "Car": 2}

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

CUSTOMER_CLUSTER_INFO = {
    0: {"name":"Value Regular Customer", "desc":"Pelanggan dengan transaksi relatif stabil dan pola order normal.", "rec":"Berikan voucher ringan, loyalty points, dan rekomendasi menu populer agar repeat order meningkat."},
    1: {"name":"High Value Customer", "desc":"Pelanggan bernilai tinggi dengan total belanja atau kuantitas lebih besar.", "rec":"Prioritaskan premium promo, bundle hemat, subscription, dan personalized campaign."},
    2: {"name":"Quick Small Order Customer", "desc":"Pelanggan dengan pembelian kecil dan cenderung praktis/cepat.", "rec":"Tawarkan promo ongkir, quick reorder, dan rekomendasi makanan cepat saji."},
    3: {"name":"Route-Sensitive Customer", "desc":"Pelanggan yang transaksinya lebih dipengaruhi rute/traffic/waktu order.", "rec":"Optimalkan estimasi waktu, pilih driver terdekat, dan tampilkan transparansi delivery fee."},
}
PERF_CLUSTER_INFO = {
    0: {"name":"Efficient Delivery", "desc":"Durasi dan rute relatif efisien.", "insight":"Pengiriman termasuk baik. Pertahankan alokasi driver dan estimasi waktu seperti saat ini."},
    1: {"name":"Moderate Delivery", "desc":"Performa normal, tetapi masih punya ruang optimasi.", "insight":"Masih cukup efisien, namun perlu monitoring pada jam sibuk dan area rute panjang."},
    2: {"name":"Traffic/Route Heavy Delivery", "desc":"Pengiriman cenderung terdampak traffic atau rute yang lebih berat.", "insight":"Kurang efisien jika sering terjadi. Perlu route optimization dan penempatan driver lebih strategis."},
    3: {"name":"Slow Delivery Risk", "desc":"Durasi tinggi atau kombinasi faktor pengiriman kurang ideal.", "insight":"Perlu perhatian khusus: cek traffic tinggi, kendaraan, jam delivery, dan estimasi waktu ke customer."},
}

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
    missing = [name for name in required.values() if not (MODEL_DIR / name).exists()]
    if missing:
        st.error("File model belum lengkap. Jalankan `python model_training.py` terlebih dahulu.")
        st.stop()
    return {key: joblib.load(MODEL_DIR / filename) for key, filename in required.items()}

@st.cache_data
def load_sample_data():
    path = DATA_DIR / "sample_talabat.csv"
    if path.exists():
        return pd.read_csv(path)
    rng = np.random.default_rng(42)
    n = 500
    df = pd.DataFrame({
        "Total_Price": rng.gamma(3, 25, n).round(2),
        "Quantity": rng.integers(1, 8, n),
        "Order_Hour": rng.integers(0, 24, n),
        "Delivery_Hour": rng.integers(0, 24, n),
        "Delivery_Duration_Minutes": np.clip(rng.normal(38, 10, n), 10, 90).round(1),
        "Total_Delivery_Route": np.clip(rng.normal(8, 3, n), 1, 22).round(2),
        "Traffic_Level": rng.choice(["Low", "Medium", "High"], n, p=[.35,.45,.20]),
        "Payment_Method": rng.choice(["Cash", "Card", "Wallet"], n),
        "Driver_Vehicle": rng.choice(["Bicycle", "Bike", "Car"], n, p=[.20,.60,.20]),
    })
    return df

def hero(title, subtitle):
    st.markdown(f"""<div class='hero'><h1>{title}</h1><p>{subtitle}</p></div>""", unsafe_allow_html=True)

def metric_card(label, value):
    st.markdown(f"<div class='small-card'><div class='metric-label'>{label}</div><div class='metric-value'>{value}</div></div>", unsafe_allow_html=True)

def transform_numeric(value, mean, std):
    return (value - mean) / std if std else 0

def predict_customer(models, total_price, quantity, order_hour, payment, traffic, route):
    # Notebook memakai fitur *_std untuk Total Price, Quantity, dan Order Hour.
    # Mean/std berikut disimpan di scaler model_training sebagai training statistics.
    stats = getattr(models["customer_scaler"], "raw_stats_", {})
    row = pd.DataFrame([{
        "Total_Price_std": transform_numeric(total_price, stats.get("Total_Price_mean", 75), stats.get("Total_Price_std", 45)),
        "Quantity_std": transform_numeric(quantity, stats.get("Quantity_mean", 3), stats.get("Quantity_std", 1.8)),
        "Order_Hour_std": transform_numeric(order_hour, stats.get("Order_Hour_mean", 13), stats.get("Order_Hour_std", 6)),
        "Payment_Method_Encoded": PAYMENT_MAP[payment],
        "Traffic_Level_Encoded": TRAFFIC_MAP[traffic],
        "Total_Delivery_Route": route,
    }], columns=CUSTOMER_FEATURES)
    scaled = models["customer_scaler"].transform(row)
    pca_point = models["customer_pca"].transform(scaled)
    cluster = int(models["customer_kmeans"].predict(pca_point)[0])
    return cluster, pca_point

def predict_performance(models, duration, route, traffic, vehicle, delivery_hour):
    stats = getattr(models["performance_scaler"], "raw_stats_", {})
    row = pd.DataFrame([{
        "Delivery_Duration_Minutes_std": transform_numeric(duration, stats.get("Delivery_Duration_Minutes_mean", 38), stats.get("Delivery_Duration_Minutes_std", 10)),
        "Total_Delivery_Route": route,
        "Traffic_Level_Encoded": TRAFFIC_MAP[traffic],
        "Driver_Vehicle_Encoded": VEHICLE_MAP[vehicle],
        "Delivery_Hour_std": transform_numeric(delivery_hour, stats.get("Delivery_Hour_mean", 13), stats.get("Delivery_Hour_std", 6)),
    }], columns=PERFORMANCE_FEATURES)
    scaled = models["performance_scaler"].transform(row)
    pca_point = models["performance_pca"].transform(scaled)
    cluster = int(models["performance_kmeans"].predict(pca_point)[0])
    return cluster, pca_point

def home_page(df):
    hero("🛵 Talabat Delivery Intelligence Dashboard", "Dashboard interaktif untuk customer segmentation dan delivery performance clustering menggunakan K-Means sebagai model utama. Dibuat dengan Streamlit, Plotly, PCA, dan machine learning pipeline yang siap deploy.")
    c1,c2,c3,c4 = st.columns(4)
    with c1: metric_card("Total Data", f"{len(df):,}")
    with c2: metric_card("Jumlah Fitur Utama", "11")
    with c3: metric_card("Best Model", "K-Means")
    with c4: metric_card("Best Silhouette", "0.481242")
    st.markdown("### 🎯 Fokus Project")
    st.markdown("""
    <div class='card'>
    Aplikasi ini membantu membaca pola pelanggan dan performa pengiriman. Customer segmentation digunakan untuk memahami tipe customer, sedangkan delivery performance clustering digunakan untuk menilai apakah proses pengiriman sudah efisien atau perlu optimasi.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### 📌 Regression Note")
    st.markdown("<div class='warning-box'>⚠️ XGBoost Tuned Regression hanya dijadikan fitur tambahan karena R² = <b>0.1099</b>. Artinya model regression belum kuat untuk prediksi durasi secara akurat.</div>", unsafe_allow_html=True)

def customer_page(models):
    hero("👥 Customer Segmentation", "Masukkan data transaksi customer. Sistem akan melakukan encoding → scaling → PCA transform → K-Means prediction.")
    left, right = st.columns([1,1.15])
    with left:
        st.markdown("### Input Customer")
        total_price = st.number_input("Total Price", 1.0, 1000.0, 75.0, 1.0)
        quantity = st.number_input("Quantity", 1, 50, 3)
        order_hour = st.slider("Order Hour", 0, 23, 13)
        payment = st.selectbox("Payment Method", ["Cash", "Card", "Wallet"])
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"])
        route = st.number_input("Total Delivery Route (km)", 0.1, 100.0, 8.0, 0.1)
        run = st.button("Predict Customer Cluster 🚀", use_container_width=True)
    with right:
        if run:
            cluster, pca_point = predict_customer(models,total_price,quantity,order_hour,payment,traffic,route)
            info = CUSTOMER_CLUSTER_INFO.get(cluster, CUSTOMER_CLUSTER_INFO[0])
            st.markdown(f"<div class='card'><span class='badge'>Cluster {cluster}</span><h2>{info['name']}</h2><p>{info['desc']}</p><hr><b>Business Recommendation:</b><br>{info['rec']}</div>", unsafe_allow_html=True)
            fig = go.Figure()
            centers = models["customer_kmeans"].cluster_centers_
            fig.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1], mode='markers+text', text=[f'C{i}' for i in range(len(centers))], marker=dict(size=18), name='Centroids'))
            fig.add_trace(go.Scatter(x=[pca_point[0,0]], y=[pca_point[0,1]], mode='markers', marker=dict(size=22), name='New Input'))
            fig.update_layout(title="PCA Position vs K-Means Centroids", template="plotly_white", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Isi input di kiri, lalu klik tombol predict.")

def performance_page(models):
    hero("⚡ Delivery Performance Clustering", "Analisis apakah pengiriman tergolong efisien, sedang, terdampak traffic/rute, atau berisiko lambat.")
    left, right = st.columns([1,1.15])
    with left:
        duration = st.number_input("Delivery Duration Minutes", 1.0, 180.0, 38.0, 0.5)
        route = st.number_input("Total Delivery Route (km)", 0.1, 100.0, 8.0, 0.1)
        traffic = st.selectbox("Traffic Level", ["Low", "Medium", "High"], key="perf_traffic")
        vehicle = st.selectbox("Driver Vehicle", ["Bicycle", "Bike", "Car"])
        delivery_hour = st.slider("Delivery Hour", 0, 23, 14)
        run = st.button("Predict Performance Cluster ⚡", use_container_width=True)
    with right:
        if run:
            cluster, pca_point = predict_performance(models,duration,route,traffic,vehicle,delivery_hour)
            info = PERF_CLUSTER_INFO.get(cluster, PERF_CLUSTER_INFO[0])
            box = "success-box" if cluster in [0,1] else "warning-box"
            st.markdown(f"<div class='{box}'><span class='badge'>Performance Cluster {cluster}</span><h2>{info['name']}</h2><p>{info['desc']}</p><hr><b>Insight:</b><br>{info['insight']}</div>", unsafe_allow_html=True)
            fig = go.Figure()
            centers = models["performance_kmeans"].cluster_centers_
            fig.add_trace(go.Scatter(x=centers[:,0], y=centers[:,1], mode='markers+text', text=[f'C{i}' for i in range(len(centers))], marker=dict(size=18), name='Centroids'))
            fig.add_trace(go.Scatter(x=[pca_point[0,0]], y=[pca_point[0,1]], mode='markers', marker=dict(size=22), name='New Input'))
            fig.update_layout(title="PCA Position vs Performance Centroids", template="plotly_white", height=420)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Masukkan kondisi pengiriman lalu klik predict.")

def evaluation_page():
    hero("📊 Model Evaluation", "Perbandingan K-Means, GMM, dan DBSCAN berdasarkan hasil evaluasi notebook.")
    tab1, tab2 = st.tabs(["Customer Segmentation", "Delivery Performance"])
    for tab, df_eval, title in [(tab1, CUSTOMER_EVAL, "Customer"), (tab2, PERF_EVAL, "Performance")]:
        with tab:
            st.dataframe(df_eval, use_container_width=True)
            fig1 = px.bar(df_eval, x="Model", y="Silhouette Score", text="Silhouette Score", title=f"{title} - Silhouette Score", template="plotly_white")
            fig1.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig1, use_container_width=True)
            dbi_df = df_eval.dropna(subset=["Davies-Bouldin Index"])
            fig2 = px.bar(dbi_df, x="Model", y="Davies-Bouldin Index", text="Davies-Bouldin Index", title=f"{title} - Davies-Bouldin Index", template="plotly_white")
            fig2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            st.plotly_chart(fig2, use_container_width=True)
    st.markdown("### Kenapa K-Means Dipilih?")
    st.markdown("<div class='card'>K-Means dipilih karena menghasilkan Silhouette Score tertinggi atau hampir sama dengan GMM, hasil cluster lebih stabil, mudah diinterpretasikan, dan cocok untuk dashboard segmentasi bisnis. DBSCAN kurang cocok karena score lebih rendah dan sensitif terhadap parameter eps/min_samples, sehingga sebagian data dapat dianggap noise.</div>", unsafe_allow_html=True)

def insight_page(df):
    hero("🔎 Data Insight", "Preview data, distribusi numerik, korelasi, dan limitasi dataset.")
    st.markdown("### Sample Dataset")
    st.dataframe(df.head(20), use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(df, x="Total_Price", nbins=35, title="Distribusi Total Price", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.histogram(df, x="Delivery_Duration_Minutes", nbins=35, title="Distribusi Delivery Duration", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    numeric_cols = ["Total_Price", "Quantity", "Order_Hour", "Delivery_Hour", "Delivery_Duration_Minutes", "Total_Delivery_Route"]
    corr = df[numeric_cols].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=True, title="Correlation Heatmap", template="plotly_white", aspect="auto")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Limitasi Dataset")
    st.markdown("""
    <div class='warning-box'>
    <b>1.</b> Korelasi fitur terhadap delivery duration cenderung lemah, sehingga regression memiliki R² rendah.<br>
    <b>2.</b> Clustering menggunakan PCA 2D, sehingga sebagian informasi dari fitur asli dapat berkurang.<br>
    <b>3.</b> Dataset belum tentu menangkap faktor eksternal seperti cuaca, promo, kepadatan restoran, dan real-time traffic.<br>
    <b>4.</b> Label cluster bukan label aktual, melainkan hasil pengelompokan unsupervised sehingga interpretasinya perlu validasi bisnis.
    </div>
    """, unsafe_allow_html=True)

def about_page():
    hero("ℹ️ About", "Ringkasan metode, kesimpulan, dan batasan aplikasi.")
    st.markdown("""
    <div class='card'>
    <h3>Metode</h3>
    Aplikasi menggunakan K-Means untuk dua task utama: customer segmentation dan delivery performance clustering. Input user diproses melalui encoding, standardization, PCA transform, lalu prediksi cluster.
    <h3>Kesimpulan</h3>
    K-Means menjadi model utama karena evaluasi clustering paling kuat dan interpretasinya paling cocok untuk dashboard bisnis.
    <h3>Limitasi</h3>
    Regression XGBoost tuned hanya tambahan karena R² 0.1099 masih rendah. Aplikasi ini lebih tepat dipakai sebagai decision support dan exploratory dashboard, bukan sistem prediksi durasi yang sepenuhnya akurat.
    </div>
    """, unsafe_allow_html=True)

models = load_models()
df = load_sample_data()

st.sidebar.markdown("# 🧡 Talabat ML")
page = st.sidebar.radio("Navigation", ["Home", "Customer Segmentation", "Delivery Performance", "Model Evaluation", "Data Insight", "About"])
st.sidebar.markdown("---")
st.sidebar.caption("Built for Streamlit Community Cloud")

if page == "Home": home_page(df)
elif page == "Customer Segmentation": customer_page(models)
elif page == "Delivery Performance": performance_page(models)
elif page == "Model Evaluation": evaluation_page()
elif page == "Data Insight": insight_page(df)
else: about_page()
