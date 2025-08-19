import os
import json
import joblib
import numpy as np
from lime.lime_tabular import LimeTabularExplainer

def load_artifacts():
    def pick(p):
        for c in [p, os.path.join(os.getcwd(), p)]:
            if os.path.exists(c):
                return c
        return p
    model_path = pick("data/sme_strategy_model.pkl")
    enc_path   = pick("data/label_encoder.pkl")
    cols_path  = pick("data/feature_columns.json")
    model = joblib.load(model_path)
    label_enc = joblib.load(enc_path)
    with open(cols_path, "r") as f:
        raw_cols = json.load(f)
        feature_cols = [c.strip() for c in raw_cols]
    return model, label_enc, feature_cols

def get_explainer(_model, feature_cols, label_enc):
    explainer = LimeTabularExplainer(
        training_data=np.zeros((1, len(feature_cols))),
        feature_names=feature_cols,
        class_names=list(label_enc.classes_),
        discretize_continuous=True
    )
    return explainer

def options_from_prefix(prefix, feature_cols):
    options = set()
    for col in feature_cols:
        if col.startswith(prefix):
            option = col[len(prefix):].strip()
            options.add(option)
    return sorted(options)

def set_one_hot(row, prefix, selected_value, all_options):
    selected_clean = selected_value.lower().strip().replace(" ", "").replace("-", "")
    for option in all_options:
        option_clean = option.lower().strip().replace(" ", "").replace("-", "")
        if option_clean == selected_clean:
            col_name = f"{prefix}{option}"
            if col_name in row:
                row[col_name] = 1
                return
    for option in all_options:
        option_clean = option.lower().strip().replace(" ", "").replace("-", "")
        if option_clean in selected_clean or selected_clean in option_clean:
            col_name = f"{prefix}{option}"
            if col_name in row:
                row[col_name] = 1
                return
    if all_options:
        col_name = f"{prefix}{all_options[-1]}"
        if col_name in row:
            row[col_name] = 1