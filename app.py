# app.py
import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st


# -------------------------
# Download & load helpers
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.content

def load_model_from_bytes(data: bytes) -> Any:
    # Try joblib first (typical for sklearn/xgb/LGBM dumps)
    try:
        import joblib
        return joblib.load(io.BytesIO(data))
    except Exception:
        pass
    # Fallback: pickle
    return pickle.loads(data)

def load_model_from_url(url: str) -> Any:
    data = fetch_bytes(url)
    return load_model_from_bytes(data)

def load_threshold_from_url(url: str) -> Optional[float]:
    try:
        raw = fetch_bytes(url).decode("utf-8")
        js = json.loads(raw)
        # allow {"threshold": 0.6} or {"positive_class_threshold": 0.6} or {"thresholds":{"1":0.6}}
        if isinstance(js, dict):
            if "threshold" in js and isinstance(js["threshold"], (int, float)):
                return float(js["threshold"])
            if "positive_class_threshold" in js and isinstance(js["positive_class_threshold"], (int, float)):
                return float(js["positive_class_threshold"])
            if "thresholds" in js and isinstance(js["thresholds"], dict):
                if "1" in js["thresholds"] and isinstance(js["thresholds"]["1"], (int, float)):
                    return float(js["thresholds"]["1"])
                # first numeric value
                for v in js["thresholds"].values():
                    if isinstance(v, (int, float)):
                        return float(v)
        return None
    except Exception:
        return None

def read_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    # last resort: try CSV
    return pd.read_csv(uploaded_file)

def predict_proba_safe(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return positive-class probabilities if possible, else None."""
    # predict_proba
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            # Prefer class label 1 if present; else last column
            if hasattr(model, "classes_") and 1 in getattr(model, "classes_", []):
                pos_idx = list(model.classes_).index(1)
            else:
                pos_idx = -1
            return proba[:, pos_idx]
    # decision_function -> sigmoid
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if np.ndim(s) == 1:
            return 1.0 / (1.0 + np.exp(-s))
    return None


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="GitHub Model Batch Predictor", page_icon="🧪", layout="centered")
st.title("🧪 Simple Batch Predictor (GitHub model)")
st.caption("Paste RAW GitHub URLs for your model & threshold. Upload a DataFrame. Get probabilities and labels.")

with st.sidebar:
    st.header("1) Model from GitHub")
    gh_model_url = st.text_input(
        "RAW URL to xgb_mid_model.joblib",
        placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/xgb_mid_model.joblib",
    )
    st.header("2) Threshold (optional)")
    gh_thr_url = st.text_input(
        "RAW URL to xgb_mid_threshold.json",
        placeholder="https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/xgb_mid_threshold.json",
    )

    st.header("3) Data")
    data_file = st.file_uploader("Upload DataFrame (.csv / .parquet)", type=["csv", "parquet", "pq"])

    st.header("Options")
    default_threshold = st.number_input("Decision threshold (fallback)", 0.0, 1.0, 0.5, 0.01)
    run = st.button("Predict")

# Load model
model = None
thr = None

if gh_model_url.strip():
    try:
        model = load_model_from_url(gh_model_url.strip())
        st.success("Model loaded from GitHub.")
    except Exception as e:
        st.error(f"Could not load model from GitHub: {e}")

if gh_thr_url.strip():
    val = load_threshold_from_url(gh_thr_url.strip())
    if val is not None:
        thr = float(val)
        st.info(f"Threshold from JSON: {thr:.4f}")
    else:
        st.warning("Could not parse threshold JSON; using fallback.")

# Load data
df = None
if data_file is not None:
    try:
        df = read_dataframe(data_file)
        st.success(f"Loaded data: {data_file.name}  •  {len(df):,} rows, {df.shape[1]} cols")
        st.write("Preview:")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read data: {e}")

# Feature selection
feat_cols, id_col = [], "(none)"
if df is not None:
    st.subheader("Select feature columns")
    # by default, everything except obvious labels
    default_feats = [c for c in df.columns if c.lower() not in {"class", "target", "label"}]
    feat_cols = st.multiselect("Features used by the model", options=list(df.columns),
                               default=default_feats if default_feats else list(df.columns))
    id_col = st.selectbox("Optional ID column to keep", options=["(none)"] + list(df.columns), index=0)

# Predict
if run:
    if model is None:
        st.warning("Provide a valid RAW GitHub URL for the model.")
    elif df is None:
        st.warning("Upload a data file first.")
    elif not feat_cols:
        st.warning("Select at least one feature column.")
    else:
        X = df[feat_cols].copy()

        # Auto-align to model.feature_names_in_ if present
        if hasattr(model, "feature_names_in_"):
            needed = list(model.feature_names_in_)
            missing = [c for c in needed if c not in X.columns]
            extra = [c for c in X.columns if c not in needed]
            for m in missing:
                X[m] = 0.0
            X = X[needed]
            if missing:
                st.info(f"Added {len(missing)} missing feature(s) as 0: {missing[:8]}{' ...' if len(missing)>8 else ''}")
            if extra:
                st.info(f"Ignored {len(extra)} extra column(s) not used by the model: {extra[:8]}{' ...' if len(extra)>8 else ''}")

        # Probabilities if possible
        y_prob = predict_proba_safe(model, X)
        if y_prob is None:
            # Fall back to labels only
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
                    file_name=f"predictions_{Path('github_model').stem}.csv",
                    mime="text/csv",
                )
            else:
                st.error("Model has neither predict_proba/decision_function nor predict.")
        else:
            # Use JSON threshold if provided; else fallback from sidebar
            use_thr = float(thr) if thr is not None else float(default_threshold)
            y_pred = (y_prob >= use_thr).astype(int)

            out = pd.DataFrame(index=df.index)
            if id_col != "(none)" and id_col in df.columns:
                out[id_col] = df[id_col]
            out["probability"] = y_prob
            out["prediction"] = y_pred

            st.subheader("Results")
            st.dataframe(out.head(30), use_container_width=True)

            st.caption("Probability distribution (binned):")
            hist = pd.Series(y_prob).value_counts(bins=20, sort=False)
            st.bar_chart(hist)

            st.download_button(
                "💾 Download results (CSV)",
                out.to_csv(index=False).encode("utf-8"),
                file_name=f"predictions_{Path('github_model').stem}.csv",
                mime="text/csv",
            )

st.caption("Tip: Use the **RAW** GitHub file URL (button on GitHub file view).")
