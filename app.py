"""
Sydney Housing Price Predictor - Streamlit web app (Part 5).
Run from this folder with:   streamlit run app.py
Needs model.joblib and model_metadata.json (created by running the notebook) and housing_features.py.
"""
import json
from datetime import date

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import housing_features as hf

st.set_page_config(page_title="Sydney Housing Price Predictor", page_icon="🏠", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("model.joblib")
    with open("model_metadata.json") as f:
        meta = json.load(f)
    return model, meta


try:
    model, meta = load_artifacts()
except FileNotFoundError:
    st.error("model.joblib / model_metadata.json not found. Run the notebook first to train and save the model.")
    st.stop()

used = set(meta["inputs_used"])
ranges = meta["ranges"]
fmt = lambda v: f"${v:,.0f}"

st.title("🏠 Sydney Housing Price Predictor")
st.caption(f"Model: {meta['model_name']} trained on {meta['n_training_rows']} sold properties in "
           f"{', '.join(meta['suburbs'])} (sales up to {meta['latest_sale_date']}).")


def range_warnings(row: dict):
    """Warn when an input lies outside the range the model has seen (predictions are less reliable)."""
    msgs = []
    for col, (lo, hi) in ranges.items():
        v = row.get(col)
        if v is not None and not pd.isna(v) and (v < lo or v > hi):
            msgs.append(f"{col.replace('_', ' ')} = {v:g} is outside the training range ({lo:g} - {hi:g}).")
    return msgs


def show_prediction(pred, low, high, row):
    st.success("Estimated sale price")
    c1, c2 = st.columns(2)
    c1.metric("Predicted price", fmt(pred))
    c2.metric("Likely range (approx. 80%)", f"{fmt(low)} - {fmt(high)}")
    st.caption(f"The range comes from how far out-of-sample predictions were from actual prices during testing. "
               f"Typical error of this model: about {fmt(meta['cv_mae'])} (MAE), {meta['cv_mape_pct']:.0f}% on average.")
    for m in range_warnings(row):
        st.warning(m)
    st.info("This is a statistical estimate from a small dataset. It is not a valuation and does not know about condition, "
            "street position, renovations not mentioned in the text, or current market conditions.")


tab_single, tab_batch = st.tabs(["Single property", "Upload a CSV"])

with tab_single:
    with st.form("property_form"):
        c1, c2 = st.columns(2)
        row = {}
        row["suburb"] = c1.selectbox("Suburb", meta["suburbs"])
        if meta["property_types"]:
            row["property_type"] = c2.selectbox("Property type", meta["property_types"])
        row["bedrooms"] = c1.number_input("Bedrooms", 0, 15, 3)
        row["bathrooms"] = c2.number_input("Bathrooms", 0, 15, 2)
        row["car_spaces"] = c1.number_input("Car spaces", 0, 15, 1)
        if "land_size_sqm" in used:
            unknown = c2.checkbox("Land size unknown (e.g. unit)", value=False)
            row["land_size_sqm"] = np.nan if unknown else c2.number_input("Land size (sqm)", 0.0, 100000.0, 450.0, step=10.0)
        if "building_size_sqm" in used:
            row["building_size_sqm"] = c1.number_input("Building size (sqm)", 0.0, 5000.0, 150.0, step=5.0)
        if "year_built" in used:
            row["year_built"] = c2.number_input("Year built", 1800, date.today().year, 1990)
        if "distance_to_station_km" in used:
            row["distance_to_station_km"] = c1.number_input("Distance to nearest station (km)", 0.0, 50.0, 1.0, step=0.1)
        if "distance_to_cbd_km" in used:
            row["distance_to_cbd_km"] = c2.number_input("Distance to Sydney CBD (km)", 0.0, 100.0, 8.0, step=0.5)
        if meta["sale_methods"]:
            row["sale_method"] = c2.selectbox("Sale method", meta["sale_methods"])
        row["sale_date"] = str(st.date_input("Expected sale date", value=pd.Timestamp(meta["latest_sale_date"]).date()))
        if "description" in used:
            row["description"] = st.text_area("Agent-style description (optional)",
                                              placeholder="e.g. Renovated family home with pool, close to station...")
        submitted = st.form_submit_button("Predict price")
    if submitted:
        pred, low, high = hf.predict_with_range(model, hf.make_input_frame(row), meta["ratio_low"], meta["ratio_high"])
        show_prediction(pred[0], low[0], high[0], row)

with tab_batch:
    st.write("Upload a CSV with columns such as: " + ", ".join(hf.RAW_MODEL_COLUMNS) +
             ". Only `suburb` (must be one of the trained suburbs) and the bedroom/bathroom/car-space columns are essential.")
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        data = pd.read_csv(up)
        data.columns = [c.strip().lower().replace(" ", "_") for c in data.columns]
        unknown_sub = set(data.get("suburb", pd.Series(dtype=str)).dropna()) - set(meta["suburbs"])
        if "suburb" not in data:
            st.error("The file needs a 'suburb' column.")
        else:
            if unknown_sub:
                st.warning(f"Suburbs not in the training data ({', '.join(sorted(unknown_sub))}) - predictions for these rows are unreliable.")
            if "sale_date" not in data:
                data["sale_date"] = meta["latest_sale_date"]
            pred, low, high = hf.predict_with_range(model, data, meta["ratio_low"], meta["ratio_high"])
            out = data.assign(predicted_price=pred.round(0), range_low=low.round(0), range_high=high.round(0))
            st.dataframe(out)
            st.download_button("Download predictions (CSV)", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")

with st.expander("About this model"):
    st.write(f"Cross-validated performance: MAE {fmt(meta['cv_mae'])}, RMSE {fmt(meta['cv_rmse'])}, "
             f"MAPE {meta['cv_mape_pct']:.1f}%, R² {meta['cv_r2']:.2f}.")
    st.write("Median sale price in the data by suburb:")
    st.table(pd.DataFrame({"Median price": {k: fmt(v) for k, v in meta["median_price_by_suburb"].items()}}))
