import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Optional geopy. Kalau tidak ada koordinat lengkap, app tetap bisa jalan.
try:
    from geopy.distance import geodesic
except Exception:
    geodesic = None

st.set_page_config(
    page_title="Talabat Delivery Clustering Dashboard",
    layout="wide"
)

st.title("Talabat Delivery Clustering & Dataset Limitation Dashboard")
st.write(
    "Dashboard ini digunakan untuk menampilkan hasil clustering terbaik, membandingkan model, "
    "dan menjelaskan limitasi dataset berdasarkan hasil evaluasi."
)

# =========================
# Helper Functions
# =========================

def safe_mode(series):
    mode_val = series.mode(dropna=True)
    if len(mode_val) > 0:
        return mode_val.iloc[0]
    return np.nan


def calculate_total_delivery_route(df):
    required_cols = [
        "Driver_Lat", "Driver_Lon",
        "Restaurant_Lat", "Restaurant_Lon",
        "Customer_Lat", "Customer_Lon"
    ]

    if "Total_Delivery_Route" in df.columns:
        return df

    if all(col in df.columns for col in required_cols) and geodesic is not None:
        df["Driver_Point"] = list(zip(df["Driver_Lat"], df["Driver_Lon"]))
        df["Restaurant_Point"] = list(zip(df["Restaurant_Lat"], df["Restaurant_Lon"]))
        df["Customer_Point"] = list(zip(df["Customer_Lat"], df["Customer_Lon"]))

        df["Driver_Restaurant_Distance"] = df.apply(
            lambda row: geodesic(row["Driver_Point"], row["Restaurant_Point"]).km,
            axis=1
        )

        df["Restaurant_Customer_Distance"] = df.apply(
            lambda row: geodesic(row["Restaurant_Point"], row["Customer_Point"]).km,
            axis=1
        )

        df["Total_Delivery_Route"] = (
            df["Driver_Restaurant_Distance"] + df["Restaurant_Customer_Distance"]
        )

    elif "Delivery_Distance_km" in df.columns:
        df["Total_Delivery_Route"] = df["Delivery_Distance_km"]

    return df


def preprocess_data(df):
    df = df.copy()

    # Convert datetime columns if available
    if "Order_Time" in df.columns:
        df["Order_Time"] = pd.to_datetime(df["Order_Time"], errors="coerce")
        df["Order_Hour"] = df["Order_Time"].dt.hour
        df["Order_Month"] = df["Order_Time"].dt.month
        df["Order_Year"] = df["Order_Time"].dt.year

        df["Order_Period"] = pd.cut(
            df["Order_Hour"],
            bins=[0, 6, 12, 18, 24],
            labels=["Night", "Morning", "Afternoon", "Evening"],
            include_lowest=True
        )

    if "Delivery_Time" in df.columns:
        df["Delivery_Time"] = pd.to_datetime(df["Delivery_Time"], errors="coerce")
        df["Delivery_Hour"] = df["Delivery_Time"].dt.hour

    # Create category columns for profiling
    if "Total_Price" in df.columns and "Price_Level" not in df.columns:
        try:
            df["Price_Level"] = pd.qcut(
                df["Total_Price"],
                q=4,
                labels=["Low Price", "Medium Price", "High Price", "Very High Price"],
                duplicates="drop"
            )
        except Exception:
            df["Price_Level"] = "Unknown"

    if "Quantity" in df.columns and "Quantity_Level" not in df.columns:
        df["Quantity_Level"] = pd.cut(
            df["Quantity"],
            bins=[0, 2, 4, np.inf],
            labels=["Small Order", "Medium Order", "Large Order"],
            include_lowest=True
        )

    # Encoding categorical columns
    traffic_map = {"Low": 0, "Medium": 1, "High": 2}
    vehicle_map = {"Bicycle": 0, "Motorbike": 1, "Car": 2}
    payment_map = {"Cash": 0, "Credit Card": 1, "Wallet": 2}

    if "Traffic_Level" in df.columns and "Traffic_Level_Encoded" not in df.columns:
        df["Traffic_Level_Encoded"] = df["Traffic_Level"].map(traffic_map)

    if "Driver_Vehicle" in df.columns and "Driver_Vehicle_Encoded" not in df.columns:
        df["Driver_Vehicle_Encoded"] = df["Driver_Vehicle"].map(vehicle_map)

    if "Payment_Method" in df.columns and "Payment_Method_Encoded" not in df.columns:
        df["Payment_Method_Encoded"] = df["Payment_Method"].map(payment_map)

    if "City" in df.columns and "City_Encoded" not in df.columns:
        df["City_Encoded"] = df["City"].astype("category").cat.codes

    # Standardized columns needed by notebook
    standard_cols = [
        "Quantity", "Total_Price", "Delivery_Duration_Minutes",
        "Delivery_Distance_km", "Order_Hour", "Delivery_Hour"
    ]

    for col in standard_cols:
        std_col = f"{col}_std"
        if col in df.columns and std_col not in df.columns:
            if df[col].std() == 0 or pd.isna(df[col].std()):
                df[std_col] = 0
            else:
                df[std_col] = (df[col] - df[col].mean()) / df[col].std()

    df = calculate_total_delivery_route(df)

    return df


def evaluate_clustering(X, labels):
    labels = np.array(labels)
    unique_labels = set(labels)

    if len(unique_labels) <= 1:
        return None, None, None

    return (
        silhouette_score(X, labels),
        davies_bouldin_score(X, labels),
        calinski_harabasz_score(X, labels)
    )


def build_clustering(df, analysis_type):
    if analysis_type == "Customer Segmentation":
        features = [
            "Total_Price_std",
            "Quantity_std",
            "Order_Hour_std",
            "Payment_Method_Encoded",
            "Traffic_Level_Encoded",
            "Total_Delivery_Route"
        ]
        profile_cols = {
            "Price_Level": safe_mode,
            "Quantity_Level": safe_mode,
            "Order_Period": safe_mode,
            "Payment_Method": safe_mode,
            "Total_Price": "mean"
        }
        final_col = "Customer_Cluster"
        title = "Customer Segmentation"

    else:
        features = [
            "Delivery_Duration_Minutes_std",
            "Total_Delivery_Route",
            "Traffic_Level_Encoded",
            "Driver_Vehicle_Encoded",
            "Delivery_Hour_std"
        ]
        profile_cols = {
            "Delivery_Duration_Minutes": "mean",
            "Total_Delivery_Route": "mean",
            "Traffic_Level": safe_mode,
            "Driver_Vehicle": safe_mode,
            "Delivery_Hour": "mean"
        }
        final_col = "Performance_Cluster"
        title = "Delivery Performance"

    missing_features = [col for col in features if col not in df.columns]
    if missing_features:
        st.error(f"Kolom berikut belum ada di dataset: {missing_features}")
        st.stop()

    working_df = df.dropna(subset=features).copy()
    X = working_df[features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # K-Means as final selected model
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    working_df["KMeans_Cluster"] = kmeans.fit_predict(X_pca)

    # DBSCAN as comparison only
    dbscan = DBSCAN(eps=0.35, min_samples=10)
    working_df["DBSCAN_Cluster"] = dbscan.fit_predict(X_pca)

    # GMM as comparison only
    gmm = GaussianMixture(n_components=4, random_state=42)
    working_df["GMM_Cluster"] = gmm.fit_predict(X_pca)

    # Evaluation
    sil_kmeans, dbi_kmeans, ch_kmeans = evaluate_clustering(X_pca, working_df["KMeans_Cluster"])

    dbscan_mask = working_df["DBSCAN_Cluster"] != -1
    if dbscan_mask.sum() > 0 and len(set(working_df.loc[dbscan_mask, "DBSCAN_Cluster"])) > 1:
        sil_dbscan, dbi_dbscan, ch_dbscan = evaluate_clustering(
            X_pca[dbscan_mask],
            working_df.loc[dbscan_mask, "DBSCAN_Cluster"]
        )
    else:
        sil_dbscan, dbi_dbscan, ch_dbscan = None, None, None

    sil_gmm, dbi_gmm, ch_gmm = evaluate_clustering(X_pca, working_df["GMM_Cluster"])

    evaluation_df = pd.DataFrame({
        "Model": ["K-Means", "DBSCAN", "GMM"],
        "Silhouette Score": [sil_kmeans, sil_dbscan, sil_gmm],
        "Davies-Bouldin Index": [dbi_kmeans, dbi_dbscan, dbi_gmm],
        "Calinski-Harabasz Score": [ch_kmeans, ch_dbscan, ch_gmm]
    })

    working_df[final_col] = working_df["KMeans_Cluster"]

    # Cluster profiling
    available_profile_cols = {
        col: agg for col, agg in profile_cols.items() if col in working_df.columns
    }

    if available_profile_cols:
        profile_df = working_df.groupby(final_col).agg(available_profile_cols).reset_index()
    else:
        profile_df = working_df[[final_col]].value_counts().reset_index()

    return {
        "df": working_df,
        "features": features,
        "X_pca": X_pca,
        "pca": pca,
        "kmeans": kmeans,
        "evaluation_df": evaluation_df,
        "profile_df": profile_df,
        "final_col": final_col,
        "title": title
    }


def plot_cluster(X_pca, labels, centroids, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        alpha=0.65
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        marker="X",
        s=220,
        label="Centroids"
    )

    for i, centroid in enumerate(centroids):
        ax.text(
            centroid[0],
            centroid[1],
            f"Cluster {i}",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.colorbar(scatter, ax=ax)
    return fig


def limitation_text(evaluation_df, pca):
    kmeans_row = evaluation_df[evaluation_df["Model"] == "K-Means"].iloc[0]
    silhouette = kmeans_row["Silhouette Score"]
    total_variance = pca.explained_variance_ratio_.sum()

    notes = []

    if silhouette < 0.5:
        notes.append(
            "Silhouette Score masih di bawah 0.50, sehingga pemisahan antar cluster belum terlalu kuat."
        )
    else:
        notes.append(
            "Silhouette Score cukup baik, tetapi cluster tetap perlu dibaca sebagai segmentasi analitis, bukan kelompok yang sempurna."
        )

    if total_variance < 0.6:
        notes.append(
            f"Total explained variance PCA 2D hanya sekitar {total_variance:.2%}, sehingga visualisasi 2D belum mewakili seluruh informasi data asli."
        )

    notes.append(
        "Hasil clustering menunjukkan pola umum, tetapi masih terdapat kemungkinan overlap karena fitur operasional penting seperti cuaca, waktu persiapan restoran, detail traffic, dan ketersediaan driver tidak tersedia."
    )

    notes.append(
        "Karena itu, hasil dashboard lebih tepat digunakan untuk eksplorasi segmentasi dan evaluasi limitasi dataset, bukan untuk keputusan otomatis penuh."
    )

    return notes

# =========================
# Sidebar
# =========================

st.sidebar.header("Input Data")
uploaded_file = st.sidebar.file_uploader("Upload dataset CSV", type=["csv"])

analysis_type = st.sidebar.selectbox(
    "Pilih analisis clustering",
    ["Customer Segmentation", "Delivery Performance"]
)

st.sidebar.info(
    "Model final yang digunakan adalah K-Means karena pada notebook K-Means memiliki hasil paling stabil dibanding DBSCAN dan GMM."
)

# =========================
# Main App
# =========================

if uploaded_file is None:
    st.warning("Upload file dataset CSV terlebih dahulu untuk menjalankan dashboard.")
    st.write("Kolom penting yang dibutuhkan:")
    st.code(
        """
Customer Segmentation:
- Total_Price
- Quantity
- Order_Time
- Payment_Method
- Traffic_Level
- koordinat / Delivery_Distance_km

Delivery Performance:
- Delivery_Duration_Minutes
- Delivery_Time
- Traffic_Level
- Driver_Vehicle
- koordinat / Delivery_Distance_km
        """
    )
    st.stop()

raw_df = pd.read_csv(uploaded_file)
st.subheader("1. Dataset Preview")
st.dataframe(raw_df.head())

processed_df = preprocess_data(raw_df)
result = build_clustering(processed_df, analysis_type)

clustered_df = result["df"]
evaluation_df = result["evaluation_df"]
profile_df = result["profile_df"]
final_col = result["final_col"]

st.subheader("2. Selected Final Model")
st.success("Final model: K-Means Clustering dengan 4 cluster")
st.write(
    "DBSCAN dan GMM tetap dihitung sebagai pembanding, tetapi hasil final menggunakan K-Means karena lebih stabil dan mudah diinterpretasikan."
)

col1, col2, col3 = st.columns(3)
kmeans_eval = evaluation_df[evaluation_df["Model"] == "K-Means"].iloc[0]
col1.metric("Silhouette Score", f"{kmeans_eval['Silhouette Score']:.4f}")
col2.metric("Davies-Bouldin Index", f"{kmeans_eval['Davies-Bouldin Index']:.4f}")
col3.metric("Calinski-Harabasz Score", f"{kmeans_eval['Calinski-Harabasz Score']:.2f}")

st.subheader("3. Model Evaluation Comparison")
st.dataframe(evaluation_df)

st.subheader("4. PCA Cluster Visualization")
fig = plot_cluster(
    result["X_pca"],
    clustered_df[final_col],
    result["kmeans"].cluster_centers_,
    f"K-Means {result['title']} Clustering"
)
st.pyplot(fig)

st.subheader("5. Cluster Profile")
st.dataframe(profile_df)

st.subheader("6. Cluster Distribution")
distribution_df = clustered_df[final_col].value_counts().sort_index().reset_index()
distribution_df.columns = ["Cluster", "Total Data"]
st.dataframe(distribution_df)

fig2, ax2 = plt.subplots(figsize=(7, 4))
ax2.bar(distribution_df["Cluster"].astype(str), distribution_df["Total Data"])
ax2.set_title("Cluster Distribution")
ax2.set_xlabel("Cluster")
ax2.set_ylabel("Total Data")
ax2.grid(alpha=0.3)
st.pyplot(fig2)

st.subheader("7. PCA Explained Variance")
pca_df = pd.DataFrame({
    "PCA Component": ["PCA 1", "PCA 2"],
    "Explained Variance Ratio": result["pca"].explained_variance_ratio_
})
st.dataframe(pca_df)
st.write(f"Total Explained Variance: **{result['pca'].explained_variance_ratio_.sum():.2%}**")

st.subheader("8. Dataset Limitation Analysis")
for note in limitation_text(evaluation_df, result["pca"]):
    st.write(f"- {note}")

st.subheader("9. Download Result")
csv_result = clustered_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download dataset with cluster labels",
    data=csv_result,
    file_name="talabat_cluster_result.csv",
    mime="text/csv"
)
