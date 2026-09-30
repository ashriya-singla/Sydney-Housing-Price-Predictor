"""
Shared feature-engineering and model-building code for the Sydney Housing Price project.

Both the Jupyter notebook and the Streamlit app import this module, so the app applies
EXACTLY the same transformations that were used when the model was trained.
Keep this file in the same folder as the notebook, the app and model.joblib.
"""
import re
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

# ---- raw columns the model accepts (a subset of your spreadsheet columns) --------------
RAW_NUMERIC = ["bedrooms", "bathrooms", "car_spaces", "land_size_sqm",
               "building_size_sqm", "year_built", "distance_to_station_km", "distance_to_cbd_km"]
RAW_CATEGORICAL = ["suburb", "property_type", "sale_method"]
RAW_OTHER = ["sale_date", "description"]
RAW_MODEL_COLUMNS = RAW_CATEGORICAL + RAW_NUMERIC + RAW_OTHER

# ---- keyword flags extracted from the agent's description (regular expressions) -------
TEXT_FLAGS = {
    "txt_renovated":   r"renovat|updated|modern|stylish|new kitchen|refurbish|contemporary",
    "txt_pool":        r"\bpool\b|swimming",
    "txt_views":       r"\bview|harbour|waterfront|water ?view|panoram|skyline|ocean",
    "txt_development": r"development|\bDA\b|approved|subdivi|redevelop|potential|dual occupancy",
    "txt_granny":      r"granny|self-contained|studio|second dwelling|guest",
    "txt_luxury":      r"luxur|prestige|architect|designer|premium|grand",
    "txt_transport":   r"train|station|metro|light rail|bus|transport|walk to|walking distance",
}


def add_features(X, ref_date):
    """Turn raw property rows into model features. Missing raw columns are created as NaN."""
    X = X.copy()
    for c in RAW_MODEL_COLUMNS:
        if c not in X.columns:
            X[c] = np.nan
    for c in RAW_NUMERIC:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    for c in RAW_CATEGORICAL:
        X[c] = X[c].astype("object").where(X[c].notna(), "Unknown").astype(str)

    # time trend: months since the start of the data window
    dates = pd.to_datetime(X["sale_date"], errors="coerce", dayfirst=True)
    X["months_since_start"] = (dates - pd.Timestamp(ref_date)).dt.days / 30.44

    # size features (log because prices/sizes are right-skewed); flag when land size is unknown
    X["land_missing"] = X["land_size_sqm"].isna().astype(int)
    X["log_land"] = np.log1p(X["land_size_sqm"])
    X["log_building"] = np.log1p(X["building_size_sqm"])
    X["total_rooms"] = X["bedrooms"].fillna(0) + X["bathrooms"].fillna(0)
    X["property_age"] = dates.dt.year - X["year_built"]

    # text features from the agent description
    desc = X["description"].fillna("").astype(str)
    for name, pattern in TEXT_FLAGS.items():
        X[name] = desc.str.contains(pattern, flags=re.IGNORECASE, regex=True).astype(int)
    X["txt_word_count"] = desc.str.split().str.len().fillna(0)
    return X


ENGINEERED_NUMERIC = ["months_since_start", "log_land", "log_building", "total_rooms", "property_age"]
ENGINEERED_FLAGS = ["land_missing"] + list(TEXT_FLAGS.keys())


def choose_feature_sets(df_raw, ref_date, min_coverage=0.3, col_min_coverage=None):
    """Decide which columns are usable (>= min_coverage non-missing (30%; missing land size is common for units and is flagged separately)) and build 3 nested feature sets:
    A = raw features only, B = A + engineered numeric features, C = B + text keyword flags."""
    fe = add_features(df_raw, ref_date)
    # land size is structurally missing for units (strata), so it gets a lower threshold; a land_missing flag is also added
    col_min_coverage = col_min_coverage or {"land_size_sqm": 0.2, "log_land": 0.2}
    ok = lambda c: c in fe.columns and fe[c].notna().mean() >= col_min_coverage.get(c, min_coverage)
    raw_num = [c for c in ["bedrooms", "bathrooms", "car_spaces", "land_size_sqm", "building_size_sqm",
                           "year_built", "distance_to_station_km", "distance_to_cbd_km"] if ok(c)]
    cat = [c for c in RAW_CATEGORICAL if c in fe.columns and (fe[c] != "Unknown").mean() >= min_coverage]
    eng_num = [c for c in ENGINEERED_NUMERIC if ok(c)]
    # drop raw columns that engineered versions replace (avoid duplicated information in set B/C)
    raw_only = [c for c in raw_num if c not in ("year_built",)]
    return {
        "A_raw":         dict(num=raw_only, flags=[], cat=cat),
        "B_+engineered": dict(num=raw_only + eng_num, flags=["land_missing"], cat=cat),
        "C_+text_flags": dict(num=raw_only + eng_num + ["txt_word_count"], flags=ENGINEERED_FLAGS, cat=cat),
    }


def build_model(estimator, feature_set, ref_date, scale=True):
    """Full pipeline: raw DataFrame -> features -> preprocessing -> estimator, predicting price.
    The target is modelled on the log scale (prices are right-skewed) and converted back to dollars."""
    num, flags, cat = feature_set["num"], feature_set["flags"], feature_set["cat"]
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    pre = ColumnTransformer([
        ("num", Pipeline(num_steps), num),
        ("flags", "passthrough", flags),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
    ], remainder="drop")
    pipe = Pipeline([
        ("features", FunctionTransformer(add_features, kw_args={"ref_date": str(ref_date)})),
        ("prep", pre),
        ("model", estimator),
    ])
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


# ---------- helpers used by the Streamlit app --------------------------------------------
def make_input_frame(values: dict) -> pd.DataFrame:
    """One-row DataFrame with every raw column the model may expect."""
    row = {c: values.get(c, np.nan) for c in RAW_MODEL_COLUMNS}
    return pd.DataFrame([row])


def predict_with_range(model, frame: pd.DataFrame, ratio_low: float, ratio_high: float):
    """Point prediction plus an approximate range based on how far out-of-fold predictions were
    from actual prices (ratio = actual / predicted; e.g. 10th and 90th percentiles)."""
    pred = np.asarray(model.predict(frame), dtype=float)
    return pred, pred * ratio_low, pred * ratio_high
