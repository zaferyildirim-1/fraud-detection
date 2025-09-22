# app.py
import io
import json
import pickle
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st


# -------------------------
# GitHub URL handling
# -------------------------
def to_raw_github_url(url: str) -> str:
    """
    Accepts:
      - Normal GitHub URL:
        https://github.com/<user>/<repo>/blob/<branch>/path/to/file
      - RAW URL:
        https://raw.githubusercontent.com/<user>/<repo>/<branch>/path/to/file
      - Short form:
        <user>/<repo>@<branch>:path/to/file
        (branch optional -> defaults to 'main')
    Returns a RAW URL.
    """
    u = url.strip()
    if not u:
        return u
    if u.startswith("https://raw.githubusercontent.com/"):
        return u
    if u.startswith("https://github.com/"):
        # Convert /blob/ URL → raw
        # https://github.com/u/r/blob/branch/path -> https://raw.githubusercontent.com/u/r/branch/path
        parts = u.split("/")
        # parts: ['https:', '', 'github.com', user, repo, 'blob', branch, path...]
        try:
            i_blob = parts.index("blob")
            user = parts[3]
            repo = parts[4]
            branch = parts[i_blob + 1]
            path = "/".join(parts[i_blob + 2 :])
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
        except ValueError:
            # Not a blob URL; just return as-is (will likely fail)
            return u
    # Short form: user/repo@branch:path or user/repo:path  (default branch 'main')
    if "/" in u and (":" in u or "@" in u):
        left, path = u.split(":", 1) if ":" in u else (u, "")
        if "@" in left:
            user_repo, branch = left.split("@", 1)
        else:
            user_repo, branch = left, "main"
        user, repo = user_repo.split("/", 1)
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    # If none matched, return original
    return u


# -------------------------
# Download & model helpers
# -------------------------
@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    r = requests.get(url, timeout=45)
    r.raise_for_status()
    return r.content

def load_model_from_bytes(data: bytes) -> Any:
    # Try joblib first; fallback to pickle
    try:
        import joblib
        return joblib.load(io.BytesIO(data))
    except Exception:
        return pickle.loads(data)

def load_threshold_value_from_bytes(data: bytes) -> Optional[float]:
    try:
        js = json.loads(data.decode("utf-8"))
        if isinstance(js, dict):
            if isinstance(js.get("threshold"), (int, float)):
                return float(js["threshold"])
            if isinstance(js.get("positive_class_threshold"), (int, float)):
                return float(js["positive_class_threshold"])
            thrs = js.get("thresholds")
            if isinstance(thrs, dict):
                if isinstance(thrs.get("1"), (int, float)):
                    return float(thrs["1"])
                for v in thrs.values():
                    if isinstance(v, (int, float)):
                        return float(v)
    except Exception:
        pass
    return None

def read_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    # try CSV as last resort
    return pd.read_csv(uploaded_file)

def predict_proba_safe(model, X: pd.DataFrame) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2:
            # Prefer positive class label 1 if available; else last column
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


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Simple Batch Predictor (GitHub OK)", page_icon="🧪", layout="centered")
st.title("🧪 Simple Batch Predictor")
st.caption("Paste GitHub links (normal or RAW) for your model & threshold. Upload a DataFrame to score.")

with st.sidebar:
    st.header("1) Model (GitHub link or short form)")
    model_url_in = st.text_input(
        "Model URL",
        value="https://github.com/zaferyildirim-1/fraud-detection/blob/main/xgb_mid_model.joblib",
        help="Accepts normal GitHub URLs, RAW URLs, or short form like 'user/repo@main:path/to/file'.",
    )

    st.header("2) Threshold JSON (optional)")
    thr_url_in = st.text_input(
        "Threshold URL",
        value="https://github.com/zaferyildirim-1/fraud-detection/blob/main/xgb_mid_threshold.json",
    )

    st.header("3) Data")
    data_file = st.file_uploader("Upload DataFrame (.csv / .parquet)", type=["csv", "parquet", "pq"])

    st.header("Options")
    fallback_thr = st.number_input("Decision threshold (fallback)", 0.0, 1.0, 0.5, 0.01)

    run = st.button("Predict")

# Resolve URLs to RAW
model_url = to_raw_github_url(model_url_in)
thr_url = to_raw_github_url(thr_url_in) if thr_url_in.strip() else ""

# Load model & threshold
model = None
thr_val = None
if model_url.strip():
    try:
        model_bytes = fetch_bytes(model_url)
        model = load_model_from_bytes(model_bytes)
        st.success("Model loaded.")
    except Exception as e:
        st.error(f"Model download/load failed:\n{e}")

if thr_url.strip():
    try:
        thr_bytes = fetch_bytes(thr_url)
        thr_val = load_threshold_value_from_bytes(thr_bytes)
        if thr_val is not None:
            st.info(f"Threshold from JSON: {thr_val:.4f}")
        else:
            st.warning("Could not parse a numeric threshold from JSON. Will use fallback.")
    except Exception as e:
        st.warning(f"Threshold download/parse failed; using fallback.\n{e}")

# Load data
df = None
if data_file is not None:
    try:
        df = read_dataframe(data_file)
        st.success(f"Loaded data: {data_file.name}  •  {len(df):,} rows × {df.shape[1]} cols")
        st.write("Preview:")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read data: {e}")

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
    id_col = st.selectbox("Optional ID column", options=["(none)"] + list(df.columns), index=0)

# Predict
if run:
    if model is None:
        st.warning("Please provide a valid GitHub link for the model.")
    elif df is None:
        st.warning("Upload a data file first.")
    elif not feat_cols:
        st.warning("Select at least one feature column.")
    else:
        X = df[feat_cols].copy()

        # Align to model.feature_names_in_ if present
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
                st.info(f"Ignored {len(extra)} extra column(s): {extra[:8]}{' ...' if len(extra)>8 else ''}")

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
                    file_name=f"predictions_{Path('github_model').stem}.csv",
                    mime="text/csv",
                )
            else:
                st.error("Model exposes neither predict_proba/decision_function nor predict.")
        else:
            use_thr = float(thr_val) if thr_val is not None else float(fallback_thr)
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

st.caption("You can paste normal GitHub URLs. The app converts them to RAW automatically.")
