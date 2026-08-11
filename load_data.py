import os

import pandas as pd

CSV_PATH = os.path.join(os.path.dirname(__file__), "placement_predict_50k Dataset.csv")


def load_data():
    """Loads the placement dataset as a pandas DataFrame."""
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"File not found: {CSV_PATH}. Update CSV_PATH at top of script.")
    return pd.read_csv(CSV_PATH)


def get_data_summary():
    """Returns a summary dict for the Data Loading page."""
    data = load_data()

    missing = data.isnull().sum()
    missing = missing[missing > 0].to_dict()

    target_col = "PlacementStatus" if "PlacementStatus" in data.columns else data.columns[-1]

    return {
        "n_rows": len(data),
        "n_cols": len(data.columns),
        "duplicate_count": int(data.duplicated().sum()),
        "missing": missing,
        "target_counts": data[target_col].value_counts().to_dict(),
    }
