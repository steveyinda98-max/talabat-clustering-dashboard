# Talabat Delivery Intelligence Dashboard

Streamlit dashboard untuk **customer segmentation** dan **delivery performance clustering** berdasarkan notebook Talabat Data Modelling.

## Main Features

- Home dashboard dengan metric cards
- Customer Segmentation menggunakan K-Means
- Delivery Performance Clustering menggunakan K-Means
- Model Evaluation untuk K-Means, GMM, dan DBSCAN
- Data Insight: sample dataset, distribusi, correlation heatmap, dan limitasi dataset
- UI modern dengan tema orange/black/white seperti Talabat

## Model Evaluation Used

### Customer Segmentation
- K-Means Silhouette Score: `0.481242`
- K-Means Davies-Bouldin Index: `0.695353`
- K-Means Calinski-Harabasz Score: `106038.562645`
- GMM Silhouette Score: `0.480522`
- DBSCAN Silhouette Score: `0.299869`

### Delivery Performance
- K-Means Silhouette Score: `0.440408`
- K-Means Davies-Bouldin Index: `0.782154`
- K-Means Calinski-Harabasz Score: `102654.998004`
- GMM Silhouette Score: `0.440370`
- DBSCAN Silhouette Score: `0.388977`

### Regression Note
Best regression model: **XGBoost After Tuning**
- MAE: `7.889`
- RMSE: `9.478`
- R2 Score: `0.1099`

Regression hanya dijadikan fitur tambahan karena nilai R2 masih rendah.

## Project Structure

```text
.
├── app.py
├── model_training.py
├── requirements.txt
├── README.md
├── data/
│   └── sample_talabat.csv
└── models/
    ├── customer_kmeans.pkl
    ├── customer_scaler.pkl
    ├── customer_pca.pkl
    ├── performance_kmeans.pkl
    ├── performance_scaler.pkl
    └── performance_pca.pkl
```

## How to Run Locally

```bash
pip install -r requirements.txt
python model_training.py
streamlit run app.py
```

Jika punya dataset asli:

```bash
python model_training.py --csv Talabat_Dataset.csv
streamlit run app.py
```

## Deployment to Streamlit Community Cloud

1. Upload semua file ke GitHub repository.
2. Pastikan file berikut ada di repo:
   - `app.py`
   - `requirements.txt`
   - folder `models/`
   - folder `data/`
3. Buka Streamlit Community Cloud.
4. Pilih **New app**.
5. Pilih repository GitHub.
6. Main file path isi:

```text
app.py
```

7. Klik **Deploy**.

## Important Preprocessing Order

Untuk input baru, aplikasi tidak langsung predict dari raw input. Urutan yang digunakan:

```text
raw input -> encoding -> standardization -> PCA transform -> K-Means predict
```

Ini mengikuti pipeline notebook agar prediksi cluster konsisten.
