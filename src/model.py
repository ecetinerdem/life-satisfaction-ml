from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib
import json

def get_model(model_type):
    if model_type == "linear":
        return LinearRegression()
    elif model_type == "rf":
        return RandomForestRegressor(random_state=42)
    elif model_type == "xgb":
        return XGBRegressor(random_state=42)
    else:
        raise ValueError("Invalid model type")

def save_model(model, path):
    joblib.dump(model, path)

def load_model(path):
    return joblib.load(path)

def save_results_to_json(results, path):
    with open(path, "w") as f:
        json.dump(results, f, indent=4)

def load_results_from_json(path):
    with open(path, "r") as f:
        return json.load(f)