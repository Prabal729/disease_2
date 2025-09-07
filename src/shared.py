import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import time

# ---------- Theme & Page ----------

THEME_CSS = """
<style>
:root {
  --bg1: #0b0f19;
  --bg2: #101826;
  --accent: #7c3aed;
  --accent-2: #22d3ee;
  --text: #e5e7eb;
  --muted: #94a3b8;
  --card: rgba(255, 255, 255, 0.04);
  --glass: rgba(255, 255, 255, 0.08);
  --shadow: 0 10px 30px rgba(124, 58, 237, 0.25), inset 0 0 0 1px rgba(255,255,255,0.03);
}

.stApp {
  background: radial-gradient(1200px 600px at 10% -10%, #1f1147 0%, rgba(31, 17, 71, 0) 60%),
              radial-gradient(1200px 600px at 110% 10%, #0c4a6e 0%, rgba(12, 74, 110, 0) 60%),
              linear-gradient(180deg, var(--bg1) 0%, var(--bg2) 100%);
  color: var(--text);
}

.main .block-container {
  padding-top: 2.5rem;
  padding-bottom: 3rem;
  max-width: 1200px;
}

h1.neon {
  font-weight: 800;
  letter-spacing: 0.5px;
  background: linear-gradient(90deg, #a78bfa, #22d3ee 50%, #60a5fa);
  -webkit-background-clip: text;
  background-clip: text;
  color: transparent;
  text-shadow: 0 0 20px rgba(124, 58, 237, 0.35);
}

h3.section {
  color: var(--text);
  font-weight: 700;
  margin-top: 1.25rem;
  margin-bottom: 0.75rem;
}

.glass {
  background: var(--card);
  border-radius: 18px;
  padding: 1rem 1.25rem;
  border: 1px solid rgba(148, 163, 184, 0.08);
  box-shadow: var(--shadow);
}

.feature {
  background: var(--glass);
  border-radius: 12px;
  padding: 0.75rem 0.9rem;
  border: 1px solid rgba(148, 163, 184, 0.08);
  transition: transform 0.15s ease, box-shadow 0.15s ease;
}
.feature:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 16px rgba(34, 211, 238, 0.15);
}

div.stButton > button {
  background: linear-gradient(90deg, var(--accent), var(--accent-2));
  color: white;
  border: 0;
  padding: 0.65rem 1.1rem;
  border-radius: 12px;
  font-weight: 700;
  letter-spacing: 0.3px;
  box-shadow: 0 10px 24px rgba(124, 58, 237, 0.35);
  transition: transform 0.08s ease, filter 0.15s ease;
}

div.stButton > button:hover { transform: translateY(-1px); filter: brightness(1.05); }
div.stButton > button:active { transform: translateY(0); }

.result-card .title { font-weight: 800; font-size: 1.1rem; color: var(--text); }
.result-card .pill {
  display: inline-block; padding: 0.25rem 0.6rem; border-radius: 999px;
  background: rgba(34, 211, 238, 0.12); color: #a5f3fc; font-weight: 700;
  border: 1px solid rgba(34, 211, 238, 0.25);
}

.confidence { position: relative; height: 10px; border-radius: 999px; background: rgba(148, 163, 184, 0.22); overflow: hidden; border: 1px solid rgba(148,163,184,0.18); }
.confidence .meter { height: 100%; width: 0%; background: linear-gradient(90deg, #22d3ee, #a78bfa); box-shadow: 0 0 20px rgba(34, 211, 238, 0.35); animation: grow 900ms ease forwards; }
@keyframes grow { to { width: var(--target); } }

footer { visibility: hidden; }
</style>
"""

def set_page(title: str, icon: str):
    st.set_page_config(page_title=title, page_icon=icon, layout="wide", initial_sidebar_state="expanded")

def inject_theme():
    st.markdown(THEME_CSS, unsafe_allow_html=True)

# ---------- DataFrame helpers ----------

def ensure_arrow_compatibility(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].dtype == 'object':
                try:
                    df_clean[col] = df_clean[col].astype('string')
                except Exception:
                    try:
                        df_clean[col] = df_clean[col].astype('category')
                    except Exception:
                        df_clean[col] = df_clean[col].astype(str).replace(['nan', 'None', 'NULL'], 'Unknown').astype('string')
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            try:
                if df_clean[col].dtype == 'float64':
                    df_clean[col] = df_clean[col].astype('float32')
                elif df_clean[col].dtype == 'int64':
                    df_clean[col] = df_clean[col].astype('int32')
            except Exception:
                pass
        return df_clean
    except Exception:
        return df

def safe_dataframe_display(df: pd.DataFrame, title: str = "Data", max_rows: int | None = None) -> bool:
    try:
        display_df = df.head(max_rows) if max_rows else df
        display_df = ensure_arrow_compatibility(display_df.copy())
        st.dataframe(display_df, use_container_width=True)
        return True
    except Exception as e:
        st.error(f"Error displaying dataframe '{title}': {e}")
        try:
            display_data = (df.head(max_rows) if max_rows else df).to_dict('records')
            st.json(display_data)
        except Exception as fallback_error:
            st.error(f"Fallback display also failed: {fallback_error}")
        return False

# ---------- Loaders ----------

@st.cache_resource
def load_artifacts():
    base_dir = Path(__file__).resolve().parent.parent
    candidates_model = [
        base_dir / 'models' / 'champion_model.pkl',
        Path.cwd() / 'models' / 'champion_model.pkl',
        Path('models') / 'champion_model.pkl',
        Path('D:/Portfolio/disease/models/champion_model.pkl'),
    ]
    candidates_features = [
        base_dir / 'models' / 'selected_features.pkl',
        Path.cwd() / 'models' / 'selected_features.pkl',
        Path('models') / 'selected_features.pkl',
        base_dir / 'models' / 'selected_features.csv',
        Path.cwd() / 'models' / 'selected_features.csv',
        Path('models') / 'selected_features.csv',
        Path('D:/Portfolio/disease/models/selected_features.pkl'),
        Path('D:/Portfolio/disease/models/selected_features.csv'),
    ]
    candidates_label = [
        base_dir / 'data' / 'processed' / 'label_encoder.pkl',
        Path.cwd() / 'data' / 'processed' / 'label_encoder.pkl',
        Path('data') / 'processed' / 'label_encoder.pkl',
        Path('D:/Portfolio/disease/data/processed/label_encoder.pkl'),
    ]

    model = None
    try:
        for p in candidates_model:
            if p.exists():
                model = joblib.load(p)
                break
    except Exception as e:
        st.error(f"Failed to load model: {e}")

    features = None
    for p in candidates_features:
        try:
            if p.exists() and p.suffix == '.pkl':
                features = joblib.load(p)
                break
            if p.exists() and p.suffix == '.csv':
                df_feats = pd.read_csv(p)
                features = df_feats.iloc[:, 0].tolist() if df_feats.shape[1] == 1 else df_feats.columns.tolist()
                break
        except Exception:
            continue

    label_encoder = None
    try:
        for p in candidates_label:
            if p.exists():
                label_encoder = joblib.load(p)
                break
    except Exception as e:
        st.warning(f"Could not load label encoder: {e}")

    return {"model": model, "features": features or [], "label_encoder": label_encoder}

@st.cache_data
def load_training_data(expected_features: list[str] | None):
    base_dir = Path(__file__).resolve().parent.parent
    def read_csv(path: Path):
        return pd.read_csv(path) if path.exists() else None
    X_train = read_csv(base_dir / 'data' / 'processed' / 'X_train.csv')
    y_train = read_csv(base_dir / 'data' / 'processed' / 'y_train.csv')
    X_valid = read_csv(base_dir / 'data' / 'processed' / 'X_valid.csv')
    y_valid = read_csv(base_dir / 'data' / 'processed' / 'y_valid.csv')

    def to_series(df):
        if df is None:
            return None
        return df.iloc[:, 0] if isinstance(df, pd.DataFrame) else df

    y_train_s = to_series(y_train)
    y_valid_s = to_series(y_valid)

    if expected_features:
        try:
            if isinstance(X_train, pd.DataFrame):
                X_train = X_train[[f for f in expected_features if f in X_train.columns]]
            if isinstance(X_valid, pd.DataFrame):
                X_valid = X_valid[[f for f in expected_features if f in X_valid.columns]]
        except Exception:
            pass

    try:
        if isinstance(X_train, pd.DataFrame) and isinstance(X_valid, pd.DataFrame):
            X_all = pd.concat([X_train, X_valid], ignore_index=True)
        else:
            X_all = X_train if isinstance(X_train, pd.DataFrame) else X_valid
    except Exception:
        X_all = None

    return X_train, y_train_s, X_valid, y_valid_s, X_all

# ---------- Misc ----------

def ui_toggle(label: str, value: bool = False, key: str | None = None) -> bool:
    toggle_fn = getattr(st, "toggle", None)
    if callable(toggle_fn):
        return toggle_fn(label, value=value, key=key)
    return st.checkbox(label, value=value, key=key)

def decode_labels(y, label_encoder):
    if y is None:
        return None
    if label_encoder is None:
        return y
    try:
        numeric = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
        return pd.Series(label_encoder.inverse_transform(numeric), index=y.index)
    except Exception:
        return y

