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


# ---------- GitHub helpers ----------

def to_raw_github_url(url: str) -> str:
    """Accepts normal GitHub URLs, RAW URLs, or short form and returns a RAW URL."""
    u = (url or "").strip()
    if not u:
        return u
    if u.startswith("https://raw.githubusercontent.com/"):
        return u
    if u.startswith("https://github.com/"):
        parts = u.split("/")
        try:
            i_blob = parts.index("blob")
            user = parts[3]; repo = parts[4]
            branch = parts[i_blob + 1]
            path = "/".join(parts[i_blob + 2:])
            return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
        except ValueError:
            # fallback: GitHub's /raw/ form
            # https://github.com/u/r/raw/branch/path
            try:
                i_tree = parts.index("tree")
                user = parts[3]; repo = parts[4]
                branch = parts[i_tree + 1]
                path = "/".join(parts[i_tree + 2:])
                return f"https://github.com/{user}/{repo}/raw/{branch}/{path}"
            except ValueError:
                return u
    # short form: user/repo@branch:path/to/file  (branch optional -> main)
    if "/" in u and (":" in u or "@" in u):
        left, path = u.split(":", 1) if ":" in u else (u, "")
        if "@" in left:
            user_repo, branch = left.split("@", 1)
        else:
            user_repo, branch = left, "main"
        user, repo = user_repo.split("/", 1)
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return u

@st.cache_data(show_spinner=False)
def fetch_bytes(url: str) -> bytes:
    """Download bytes with headers that discourage HTML; follow redirects."""
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "streamlit-app"
    }
    r = requests.get(url, headers=headers, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r.content

def looks_like_html(b: bytes) -> bool:
    head = b[:400].lower()
    return b and (b.startswith(b"\n") or b.startswith(b"\r\n") or b"<!doctype html" in head or b"<html" in head)

def looks_like_gitlfs_pointer(b: bytes) -> bool:
    # Git LFS pointer is tiny text like:
    # version https://git-lfs.github.com/spec/v1
    # oid sha256:...
    # size 12345
    try:
        s = b.decode("utf-8", errors="ignore")
    except Exception:
        return False
    return "git-lfs.github.com/spec/v1" in s and "oid sha256:" in s and "size " in s


# ---------- Model & data helpers ----------

def load_model_from_bytes(data: bytes) -> Any:
    """Try joblib first; fallback to pickle."""
    # joblib
    try:
        import joblib
        return joblib.load(io.BytesIO(data))
    except Exception:
        pass
    # pickle
    return pickle.loads(data)

def read_dataframe(uploaded_file) -> pd.DataFrame:
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if name.endswith(".parquet") or name.endswith(".pq"):
        return pd.read_parquet(uploaded_file)
    # Try CSV by default
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


# ---------- UI ----------

st.set_page_config(page_title="Simple Batch Predictor", page_icon="🧪", layout="centered")
st.title("🧪 Simple Batch Predictor")
st.caption("Use a GitHub link **or** upload your model file. Upload a DataFrame. Get probabilities (if available) + predictions.")

with st.sidebar:
    st.header("Model source")
    model_url_in = st.text_input(
        "GitHub URL (normal or RAW) or short form",
        value="https://github.com/zaferyildirim-1/fraud-detection/blob/main/xgb_mid_model.joblib",
        help="Examples: normal GitHub URL, RAW URL, or 'user/repo@main:path/to/file'. Leave empty if uploading locally."
    )
    model_upload = st.file_uploader("OR upload model (.joblib / .pkl)", type=["joblib", "pkl"])

    st.header("Threshold (optional)")
    thr_url_in = st.text_input(
        "GitHub URL for JSON (optional)",
        value="https://github.com/zaferyildirim-1/fraud-detection/blob/main/xgb_mid_threshold.json",
    )

    st.header("Data")
    data_file = st.file_uploader("Upload DataFrame (.csv / .parquet)", type=["csv", "parquet", "pq"])

    st.header("Options")
    fallback_thr = st.number_input("Decision threshold (fallback)", 0.0, 1.0, 0.5, 0.01)
    run = st.button("Predict")


# ---------- Load model ----------

model = None
model_source = ""
if model_upload is not None:
    try:
        model = load_model_from_bytes(model_upload.read())
        model_source = f"(local upload: {model_upload.name})"
        st.success(f"Model loaded {model_source}")
    except Exception as e:
        st.error(f"Could not load uploaded model: {e}")

elif model_url_in.strip():
    try:
        model_url = to_raw_github_url(model_url_in.strip())
        b = fetch_bytes(model_url)
        if looks_like_html(b):
            st.error("Downloaded HTML instead of a binary model. Make sure you used a RAW link.")
        elif looks_like_gitlfs_pointer(b):
            st.error(
                "This appears to be a **Git LFS pointer** (tiny text file) rather than the real model.\n\n"
                "Fix: On GitHub, click **Download raw file** for the model, then paste that URL here, "
                "or download the file to your computer and use **local upload** above."
            )
        else:
            model = load_model_from_bytes(b)
            model_source = "(downloaded from GitHub)"
            st.success(f"Model loaded {model_source}")
    except Exception as e:
        st.error(f"Model download/load failed: {e}\n\n"
                 "Tip: If the file uses Git LFS, use the **Download raw file** button and paste that URL, "
                 "or upload the file locally here.")

# ---------- Load threshold ----------

thr_val = None
if thr_url_in.strip():
    try:
        thr_url = to_raw_github_url(thr_url_in.strip())
        tb = fetch_bytes(thr_url)
        if looks_like_html(tb) or looks_like_gitlfs_pointer(tb):
            st.warning("Threshold URL did not return JSON (got HTML or LFS pointer). Falling back to sidebar value.")
        else:
            js = json.loads(tb.decode("utf-8"))
            if isinstance(js, dict):
                if isinstance(js.get("threshold"), (int, float)):
                    thr_val = float(js["threshold"])
                elif isinstance(js.get("positive_class_threshold"), (int, float)):
                    thr_val = float(js["positive_class_threshold"])
                elif isinstance(js.get("thresholds"), dict):
                    if isinstance(js["thresholds"].get("1"), (int, float)):
                        thr_val = float(js["thresholds"]["1"])
                    else:
                        # first numeric in thresholds
                        for v in js["thresholds"].values():
                            if isinstance(v, (int, float)):
                                thr_val = float(v); break
            if thr_val is not None:
                st.info(f"Threshold from JSON: {thr_val:.4f}")
            else:
                st.warning("Could not parse a numeric threshold from the JSON. Using fallback.")
    except Exception as e:
        st.warning(f"Threshold download/parse failed; using fallback. {e}")

# ---------- Load data ----------

df = None
if data_file is not None:
    try:
        df = read_dataframe(data_file)
        st.success(f"Loaded data: {data_file.name}  •  {len(df):,} rows × {df.shape[1]} cols")
        st.write("Preview:")
        st.dataframe(df.head(15), use_container_width=True)
    except Exception as e:
        st.error(f"Could not read data: {e}")

# ---------- Feature selection ----------

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

# ---------- Predict ----------

if run:
    if model is None:
        st.warning("Provide a valid model (GitHub link to RAW, or upload locally).")
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
                    file_name=f"predictions_{Path('model').stem}.csv",
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
                file_name=f"predictions_{Path('model').stem}.csv",
                mime="text/csv",
            )

st.caption("If GitHub uses **LFS** for your model, click **Download raw file** on GitHub and paste that URL, or upload the file locally here.")
