# Sydney Housing Price Prediction and Decision Support System

Predicts the sale price of a Sydney property (Bondi, Parramatta, Blacktown) from its listed features, using gradient boosting trained on 120 sold properties from realestate.com.au (April–September 2026), with a Streamlit web app for predictions.

## Files
| File | Purpose |
|---|---|
| `sydney_housing_data.xlsx` | Collected dataset (`Data` sheet): 173 sold listings recorded, of which 120 have a public sale price and are used for modelling |
| `sydney_housing_project.ipynb` | Parts 1-4: data checks, exploration, feature engineering, models, cross-validation, error analysis; trains and saves the model |
| `housing_features.py` | Feature engineering and pipeline code shared by the notebook and the app |
| `app.py` | Streamlit web app (Part 5) |
| `model.joblib`, `model_metadata.json` | Trained model and metadata, created by the notebook and loaded by the app |
| `requirements.txt` | Python packages |

## Setup (once)
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
1. Keep `sydney_housing_data.xlsx` in the same folder as the notebook. (A CSV export of the `Data` sheet named `sydney_housing_data.csv` also works and takes priority.)
2. Run `jupyter notebook sydney_housing_project.ipynb`, then choose *Kernel -> Restart & Run All* (about a minute). This recreates `model.joblib` and `model_metadata.json`.
3. Run `streamlit run app.py`. It opens at http://localhost:8501.

## Using the app
* **Single property tab:** choose suburb and property type, enter bedrooms, bathrooms, car spaces and land size (or tick "unknown"), then click **Predict price**. You get a point estimate, an approximate 80% range and warnings for inputs outside the training range.
* **Upload a CSV tab:** upload a file with columns such as `suburb`, `property_type`, `bedrooms`, `bathrooms`, `car_spaces` to predict many properties and download the results.

The prediction is a statistical estimate from a small dataset, not a valuation.
