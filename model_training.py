
"""
Model training script for Talabat Delivery Intelligence Dashboard.

Use this file when:
1. You have the original Talabat_Dataset.csv, or
2. You need to regenerate the .pkl files before deploying.

Expected pipeline based on notebook:
raw data -> feature engineering/encoding -> StandardScaler -> PCA(n_components=2) -> KMeans
"""
from pathlib import Path
import argparse
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
MODEL_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

CUSTOMER_FEATURES = [
    "Total_Price_std", "Quantity_std", "Order_Hour_std",
    "Payment_Method_Encoded", "Traffic_Level_Encoded", "Total_Delivery_Route"
]
PERFORMANCE_FEATURES = [
    "Delivery_Duration_Minutes_std", "Total_Delivery_Route",
    "Traffic_Level_Encoded", "Driver_Vehicle_Encoded", "Delivery_Hour_std"
]
PAYMENT_MAP = {"Cash": 0, "Credit Card": 1, "Card": 1, "Wallet": 2}
TRAFFIC_MAP = {"Low": 0, "Medium": 1, "High": 2}
VEHICLE_MAP = {"Bicycle": 0, "Motorbike": 1, "Bike": 1, "Car": 2}

def make_sample_dataset(n=2500, seed=42):
    """Synthetic fallback so the project can run even without the private CSV."""
    rng = np.random.default_rng(seed)
    segment = rng.choice([0,1,2,3], n, p=[.35,.25,.25,.15])
    total_price = np.where(segment==1, rng.normal(140,35,n), np.where(segment==2, rng.normal(35,12,n), rng.normal(75,25,n)))
    quantity = np.where(segment==1, rng.integers(4,10,n), rng.integers(1,5,n))
    order_hour = np.clip(rng.normal(np.where(segment==2, 11, np.where(segment==3, 20, 14)), 4, n),0,23).round().astype(int)
    route = np.clip(rng.normal(np.where(segment==3, 14, 7), 3, n),1,28)
    traffic = rng.choice(["Low","Medium","High"], n, p=[.35,.45,.20])
    vehicle = rng.choice(["Bicycle","Bike","Car"], n, p=[.2,.6,.2])
    payment = rng.choice(["Cash","Card","Wallet"], n, p=[.35,.4,.25])
    duration = np.clip(18 + route*1.7 + np.vectorize(TRAFFIC_MAP.get)(traffic)*6 + rng.normal(0,7,n), 8, 100)
    delivery_hour = np.clip(order_hour + rng.integers(0,3,n), 0, 23)
    df = pd.DataFrame({
        "Total_Price": total_price.round(2), "Quantity": quantity, "Order_Hour": order_hour,
        "Delivery_Hour": delivery_hour, "Delivery_Duration_Minutes": duration.round(1),
        "Total_Delivery_Route": route.round(2), "Traffic_Level": traffic,
        "Payment_Method": payment, "Driver_Vehicle": vehicle,
    })
    return df

def prepare_features(df):
    df = df.copy()
    if "Payment_Method" in df:
        df["Payment_Method_Encoded"] = df["Payment_Method"].map(PAYMENT_MAP).fillna(0)
    if "Traffic_Level" in df:
        df["Traffic_Level_Encoded"] = df["Traffic_Level"].map(TRAFFIC_MAP).fillna(0)
    if "Driver_Vehicle" in df:
        df["Driver_Vehicle_Encoded"] = df["Driver_Vehicle"].map(VEHICLE_MAP).fillna(1)

    # Create route if original coordinate-derived route is unavailable.
    if "Total_Delivery_Route" not in df.columns:
        if "Delivery_Distance_km" in df.columns:
            df["Total_Delivery_Route"] = df["Delivery_Distance_km"]
        else:
            df["Total_Delivery_Route"] = 8.0

    raw_cols = ["Total_Price", "Quantity", "Order_Hour", "Delivery_Duration_Minutes", "Delivery_Hour"]
    raw_stats = {}
    for col in raw_cols:
        if col in df.columns:
            mean = float(df[col].mean())
            std = float(df[col].std(ddof=0)) or 1.0
            df[col + "_std"] = (df[col] - mean) / std
            raw_stats[col + "_mean"] = mean
            raw_stats[col + "_std"] = std
    return df, raw_stats

def train_pipeline(X, n_clusters=4):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    return scaler, pca, kmeans

def main(csv_path=None):
    if csv_path and Path(csv_path).exists():
        df = pd.read_csv(csv_path)
    else:
        df = make_sample_dataset()
    df, raw_stats = prepare_features(df)
    DATA_DIR.mkdir(exist_ok=True)
    df[["Total_Price","Quantity","Order_Hour","Delivery_Hour","Delivery_Duration_Minutes","Total_Delivery_Route","Traffic_Level","Payment_Method","Driver_Vehicle"]].head(700).to_csv(DATA_DIR / "sample_talabat.csv", index=False)

    customer_scaler, customer_pca, customer_kmeans = train_pipeline(df[CUSTOMER_FEATURES], 4)
    performance_scaler, performance_pca, performance_kmeans = train_pipeline(df[PERFORMANCE_FEATURES], 4)

    # Store raw statistics inside scaler objects so app can standardize new raw input consistently.
    customer_scaler.raw_stats_ = raw_stats
    performance_scaler.raw_stats_ = raw_stats

    joblib.dump(customer_kmeans, MODEL_DIR / "customer_kmeans.pkl")
    joblib.dump(customer_scaler, MODEL_DIR / "customer_scaler.pkl")
    joblib.dump(customer_pca, MODEL_DIR / "customer_pca.pkl")
    joblib.dump(performance_kmeans, MODEL_DIR / "performance_kmeans.pkl")
    joblib.dump(performance_scaler, MODEL_DIR / "performance_scaler.pkl")
    joblib.dump(performance_pca, MODEL_DIR / "performance_pca.pkl")
    print("Model files saved to:", MODEL_DIR)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=None, help="Path to Talabat_Dataset.csv. If omitted, synthetic sample data is used.")
    args = parser.parse_args()
    main(args.csv)
