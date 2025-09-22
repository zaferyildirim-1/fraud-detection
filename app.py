# app.py
import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import streamlit as st


# -------------------------
# Small helpers
# -------------------------
def load_model(uploaded_file) -> Any:
    """Load a scikit-learn / XGBoost / LightGBM model saved via joblib or pickle."""
    data = uploaded_file.read()
    # Try joblib first (common for sklearn)
    try:
        import joblib
        return joblib.load(io.BytesIO(data))
    except Exception:
        pass
    # Fallback to pickle
    return pickle.loads(data)

def read_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    else:
        # Last-resort CSV
        return pd.read_csv(uploaded_file)

def predict_proba_safe(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return positive-class probabilities if possible, else None."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        # Binary: take column for positive class (try class 1 if present, else last col)
        if proba.ndim == 2:
            if hasattr(model, "classes_") and 1 in getattr(model, "classes_", []):
                pos_idx = list(model.classes_).index(1)
            else:
                pos_idx = -1
            return proba[:, pos_idx]
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        # Sigmoid map for binary scores
        if s.ndim == 1:
            return 1 / (1 + np.exp(-s))
    return None


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Simple Batch Predictor", page_icon="🧪", layout="centered")
st.title("🧪 Simple Batch Predictor")
st.caption("Upload a trained model and a DataFrame with feature columns. Get predictions and probabilities (if available).")

with st.sidebar:
    st.header("1) Upload model")
    model_file = st.file_uploader("Model file (.joblib / .pkl)", type=["joblib", "pkl"])

    st.header("2) Upload data")
    data_file = st.file_uploader("Data file (.csv / .parquet)", type=["csv", "parquet", "pq"])

    st.header("Options")
    default_threshold = st.number_input("Decision threshold (binary)", 0.0, 1.0, 0.5, 0.01)
    run = st.button("Predict")

# State
model = None
df = None

# Load model
if model_file is not None:
    try:
        model = load_model(model_file)
        st.success(f"Loaded model: {model_file.name}")
    except Exception as e:
        st.error(f"Could not load model: {e}")

# Load data
if data_file is not None:
    try:
        df = read_dataframe(data_file)
        st.success(f"Loaded data: {data_file.name}  •  {len(df):,} rows, {df.shape[1]} cols")
        st.write("Data preview:")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read data: {e}")

# Feature selection
if df is not None:
    st.subheader("Select feature columns")
    default_feats = [c for c in df.columns if c.lower() not in {"class", "target", "label"}]
    feat_cols = st.multiselect(
        "Features used by the model",
        options=list(df.columns),
        default=default_feats if default_feats else list(df.columns),
    )
    id_col = st.selectbox("Optional ID column to keep", options=["(none)"] + list(df.columns), index=0)
else:
    feat_cols, id_col = [], "(none)"

# Run prediction
if run:
    if model is None:
        st.warning("Upload a model first.")
    elif df is None:
        st.warning("Upload a data file first.")
    elif not feat_cols:
        st.warning("Select at least one feature column.")
    else:
        X = df[feat_cols].copy()

        # Try to align to model.feature_names_in_, if it exists
        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)
            missing = [c for c in needed if c not in X.columns]
            extra = [c for c in X.columns if c not in needed]
            # Add missing as 0.0
            for m in missing:
                X[m] = 0.0
            # Reorder to match model
            X = X[needed]
            if missing:
                st.info(f"Added {len(missing)} missing feature(s) as 0: {missing[:8]}{' ...' if len(missing)>8 else ''}")
            if extra:
                st.info(f"Ignored {len(extra)} extra column(s) not used by the model: {extra[:8]}{' ...' if len(extra)>8 else ''}")

        # Predictions
        y_prob = predict_proba_safe(model, X)
        # If probability not available, fall back to label prediction
        if y_prob is None:
            if hasattr(model, "predict"):
                y_pred = model.predict(X)
                out = pd.DataFrame(index=df.index)
                if id_col != "(none)" and id_col in df.columns:
                    out[id_col] = df[id_col]
                out["prediction"] = y_pred
                st.subheader("Results")
                st.dataframe(out.head(30), use_container_width=True)
                st.download_button(
                    "💾 Download results (CSV)",
                    out.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{Path(model_file.name).stem}.csv",
                    mime="text/csv",
                )
            else:
                st.error("Model has neither predict_proba/decision_function nor predict.")
        else:
            # Binary probabilities available
            y_pred = (y_prob >= default_threshold).astype(int)
            out = pd.DataFrame(index=df.index)
            if id_col != "(none)" and id_col in df.columns:
                out[id_col] = df[id_col]
            out["probability"] = y_prob
            out["prediction"] = y_pred

            st.subheader("Results")
            st.dataframe(out.head(30), use_container_width=True)

            # Quick distribution
            st.caption("Probability distribution (binned):")
            hist = pd.Series(y_prob).value_counts(bins=20, sort=False)
            st.bar_chart(hist)

            st.download_button(
                "💾 Download results (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{Path(model_file.name).stem}.csv",
                mime="text/csv",
            )

st.caption("Tip: If your model exposes `feature_names_in_`, features will be auto-aligned and missing ones filled with 0.")
