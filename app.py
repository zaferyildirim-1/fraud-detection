# app.py
import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st


def load_model_from_bytes(data: bytes) -> Any:
    try:
        import joblib
        return joblib.load(io.BytesIO(data))
    except Exception:
        return pickle.loads(data)

def read_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    return pd.read_csv(uploaded_file)

def predict_proba_safe(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            if hasattr(model, "classes_") and 1 in getattr(model, "classes_", []):
                pos_idx = list(model.classes_).index(1)
            else:
                pos_idx = -1
            return proba[:, pos_idx]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if np.ndim(s) == 1:
            return 1.0 / (1.0 + np.exp(-s))
    return None


st.set_page_config(page_title="Batch Predictor", page_icon="🧪", layout="centered")
st.title("🧪 Simple Batch Predictor")
st.caption("Upload your model (.joblib / .pkl) and a DataFrame. Get probabilities and predictions.")

with st.sidebar:
    model_file = st.file_uploader("Upload model (.joblib / .pkl)", type=["joblib", "pkl"])
    thr_file = st.file_uploader("Optional threshold JSON", type=["json"])
    data_file = st.file_uploader("Upload DataFrame (.csv / .parquet)", type=["csv", "parquet", "pq"])
    fallback_thr = st.number_input("Decision threshold (fallback)", 0.0, 1.0, 0.5, 0.01)
    run = st.button("Predict")

model, thr_val, df = None, None, None

if model_file is not None:
    try:
        model = load_model_from_bytes(model_file.read())
        st.success(f"Model loaded: {model_file.name}")
    except Exception as e:
        st.error(f"Could not load model: {e}")

if thr_file is not None:
    try:
        js = json.load(thr_file)
        if isinstance(js.get("threshold"), (int, float)):
            thr_val = float(js["threshold"])
            st.info(f"Threshold from JSON: {thr_val:.4f}")
    except Exception as e:
        st.warning(f"Threshold parse failed: {e}")

if data_file is not None:
    try:
        df = read_dataframe(data_file)
        st.success(f"Loaded data: {data_file.name}  •  {len(df):,} rows × {df.shape[1]} cols")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read data: {e}")

feat_cols, id_col = [], "(none)"
if df is not None:
    default_feats = [c for c in df.columns if c.lower() not in {"class", "target", "label"}]
    feat_cols = st.multiselect("Features used by the model", options=list(df.columns),
                               default=default_feats if default_feats else list(df.columns))
    id_col = st.selectbox("Optional ID column", options=["(none)"] + list(df.columns), index=0)

if run:
    if model is None:
        st.warning("Upload a model first.")
    elif df is None:
        st.warning("Upload a DataFrame first.")
    elif not feat_cols:
        st.warning("Select at least one feature column.")
    else:
        X = df[feat_cols].copy()

        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)
            missing = [c for c in needed if c not in X.columns]
            for m in missing:
                X[m] = 0.0
            X = X[needed]

        y_prob = predict_proba_safe(model, X)
        if y_prob is None:
            y_pred = model.predict(X)
            out = pd.DataFrame({"prediction": y_pred})
        else:
            use_thr = float(thr_val) if thr_val is not None else float(fallback_thr)
            y_pred = (y_prob >= use_thr).astype(int)
            out = pd.DataFrame({"probability": y_prob, "prediction": y_pred})

        if id_col != "(none)" and id_col in df.columns:
            out.insert(0, id_col, df[id_col])

        st.subheader("Results")
        st.dataframe(out.head(30), use_container_width=True)
        st.download_button("💾 Download results (CSV)",
                           out.to_csv(index=False).encode("utf-8"),
                           file_name="predictions.csv", mime="text/csv")
