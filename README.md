# Sydney Housing Price Prediction and Decision Support System

Files
| File | Purpose |
|---|---|
| `data_collection_template.xlsx` | Template + guide for building your dataset (sold properties in Bondi, Parramatta and Blacktown) |
| `sydney_housing_data.csv` | **Your collected dataset** (export the `Data` sheet as CSV). Not included until you create it |
| `sydney_housing_project.ipynb` | Parts 1-4: data checks, EDA, feature engineering, models, cross-validation, error analysis; trains and saves the model |
| `housing_features.py` | Feature engineering + pipeline code shared by the notebook and the app |
| `app.py` | Streamlit web app (Part 5) |
| `model.joblib`, `model_metadata.json` | Created by the notebook; loaded by the app |
| `requirements.txt` | Python packages |

## Setup (once)
```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run
1. Put `sydney_housing_data.csv` in this folder (columns as in the template; blanks allowed for optional columns).
2. `jupyter notebook sydney_housing_project.ipynb` -> *Kernel -> Restart & Run All* (about a minute). This creates `model.joblib` and `model_metadata.json`.
3. `streamlit run app.py` -> opens http://localhost:8501.

## Using the app
* **Single property tab:** choose suburb and property type, enter bedrooms, bathrooms, car spaces (and land size etc.), optionally paste a listing description, click **Predict price**. You get a point estimate, an approximate 80% range and warnings for inputs outside the training range.
* **Upload a CSV tab:** upload a file with the same column names to predict many properties and download the results.

The prediction is a statistical estimate from a small dataset, not a valuation.
