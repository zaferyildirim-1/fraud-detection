# app.py
from pathlib import Path
import io
import json
import pickle
from typing import Any, Optional, List

import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------
# Paths (same repo as app.py)
# -----------------------------
HERE = Path(__file__).parent
MODEL_PATH = HERE / "xgb_mid_model.joblib"
THRESHOLD_PATH = HERE / "xgb_mid_threshold.json"   # optional


# -----------------------------
# Helpers
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: Path) -> Any:
    """Load joblib/pickle model from disk."""
    data = path.read_bytes()
    # Try joblib first, fallback to pickle
    try:
        import joblib
        return joblib.load(io.BytesIO(data))
    except Exception:
        return pickle.loads(data)

@st.cache_data(show_spinner=False)
def load_threshold(path: Path) -> Optional[float]:
    """Read a numeric threshold from JSON (several shapes supported)."""
    if not path.exists():
        return None
    try:
        js = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(js, dict):
            if isinstance(js.get("threshold"), (int, float)):
                return float(js["threshold"])
            if isinstance(js.get("positive_class_threshold"), (int, float)):
                return float(js["positive_class_threshold"])
            thrs = js.get("thresholds")
            if isinstance(thrs, dict):
                if isinstance(thrs.get("1"), (int, float)):
                    return float(thrs["1"])
                # fallback to first numeric
                for v in thrs.values():
                    if isinstance(v, (int, float)):
                        return float(v)
    except Exception:
        pass
    return None

def read_csv(uploaded_file) -> pd.DataFrame:
    return pd.read_csv(uploaded_file)

def predict_proba_safe(model: Any, X: pd.DataFrame) -> Optional[np.ndarray]:
    """Return positive-class probabilities if possible, else None."""
    # predict_proba preferred
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            if hasattr(model, "classes_") and 1 in getattr(model, "classes_", []):
                pos_idx = list(model.classes_).index(1)
            else:
                pos_idx = -1  # last column
            return proba[:, pos_idx]
    # decision_function → sigmoid
    if hasattr(model, "decision_function"):
        s = model.decision_function(X)
        if np.ndim(s) == 1:
            return 1.0 / (1.0 + np.exp(-s))
    return None

def align_features(df: pd.DataFrame, model: Any) -> (pd.DataFrame, List[str], List[str]):
    """Align DF columns to model.feature_names_in_ if present; add missing as 0."""
    missing, extra = [], []
    if hasattr(model, "feature_names_in_"):
        needed = list(model.feature_names_in_)
        cols = list(df.columns)
        missing = [c for c in needed if c not in cols]
        extra = [c for c in cols if c not in needed]
        df_aligned = df.copy()
        for m in missing:
            df_aligned[m] = 0.0
        df_aligned = df_aligned[needed]  # reorder & drop extras
        return df_aligned, missing, extra
    return df, missing, extra


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Fraud – Batch Predictor", page_icon="🧮", layout="centered")
st.title("🧮 Simple Batch Predictor (repo model)")
st.caption("Uses `xgb_mid_model.joblib` and optional `xgb_mid_threshold.json` from this repository. Upload a CSV of features.")

# Load model & threshold at app start
if not MODEL_PATH.exists():
    st.error(f"Model file not found at: {MODEL_PATH.name}")
    st.stop()

try:
    model = load_model(MODEL_PATH)
    st.success(f"Model loaded: {MODEL_PATH.name}")
except Exception as e:
    st.error(f"Failed to load model `{MODEL_PATH.name}`: {e}")
    st.stop()

thr_json_val = load_threshold(THRESHOLD_PATH)
if thr_json_val is not None:
    st.info(f"Threshold from `{THRESHOLD_PATH.name}`: {thr_json_val:.4f}")

with st.sidebar:
    st.header("1) Upload features (CSV)")
    data_file = st.file_uploader("Choose CSV file", type=["csv"])
    st.header("2) Options")
    fallback_thr = st.number_input(
        "Decision threshold (fallback / override)",
        min_value=0.0, max_value=1.0,
        value=float(thr_json_val) if thr_json_val is not None else 0.5,
        step=0.01
    )
    run = st.button("Predict")

# Read data
df = None
if data_file is not None:
    try:
        df = read_csv(data_file)
        st.success(f"Loaded CSV: {data_file.name}  •  {len(df):,} rows × {df.shape[1]} cols")
        st.write("Preview:")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read CSV: {e}")

# Feature selection
feat_cols, id_col = [], "(none)"
if df is not None:
    st.subheader("Select feature columns")
    default_feats = [c for c in df.columns if c.lower() not in {"class", "target", "label"}]
    feat_cols = st.multiselect(
        "Features used by the model",
        options=list(df.columns),
        default=default_feats if default_feats else list(df.columns),
    )
    id_col = st.selectbox("Optional ID column to keep", options=["(none)"] + list(df.columns), index=0)

# Predict
if run:
    if df is None:
        st.warning("Upload a CSV first.")
    elif not feat_cols:
        st.warning("Select at least one feature column.")
    else:
        X_raw = df[feat_cols].copy()
        X, missing, extra = align_features(X_raw, model)
        if missing:
            st.info(f"Added {len(missing)} missing feature(s) as 0: {missing[:10]}{' ...' if len(missing)>10 else ''}")
        if extra and hasattr(model, 'feature_names_in_'):
            st.info(f"Ignored {len(extra)} extra column(s): {extra[:10]}{' ...' if len(extra)>10 else ''}")

        # Probabilities if possible
        y_prob = predict_proba_safe(model, X)
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
                    file_name="predictions.csv",
                    mime="text/csv",
                )
            else:
                st.error("Model has neither predict_proba/decision_function nor predict.")
        else:
            use_thr = float(fallback_thr)  # acts as override even if JSON exists
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
                file_name="predictions.csv",
                mime="text/csv",
            )

st.caption("Tip: `feature_names_in_` found → columns auto-aligned; missing features filled with 0.")
