import numpy as np
import torch
import torch.nn as nn
from itertools import product
import pickle
import os
import sqlite3
import pandas as pd

# Model class
class PostSaunaMLP(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, output_size=3):
        super(PostSaunaMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)

# Save to database
def save_recommendation(current_metrics, suggested_params, pred_post_sleep, pred_post_stress, pred_post_readiness, db_path='recommendation.db'):
    recommendation = {
        'recommended_temp': int(round(suggested_params[0])),
        'recommended_hum': int(round(suggested_params[1])),
        'recommended_duration': int(round(suggested_params[2])),
    }
    conn = sqlite3.connect(db_path)
    pd.DataFrame([recommendation]).to_sql('recommendations', conn, if_exists='replace', index=False)
    conn.close()
    return recommendation

# Load model
def load_model_and_scalers(model_path='model.pth', scaler_X_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    model = PostSaunaMLP()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with open(scaler_X_path, 'rb') as f:
        scaler_X = pickle.load(f)
    with open(scaler_y_path, 'rb') as f:
        scaler_y = pickle.load(f)

    return model, scaler_X, scaler_y

# Predict effects
def predict_effects(model, scaler_X, scaler_y, current_metrics, sauna_params):
    full_params = np.concatenate([current_metrics, sauna_params])
    params_scaled = scaler_X.transform(np.array([full_params]))
    pred_scaled = model(torch.tensor(params_scaled, dtype=torch.float32)).detach().numpy()
    pred = scaler_y.inverse_transform(pred_scaled)
    return pred[0]

# Tailor session
def tailor_sauna_session(model, scaler_X, scaler_y, current_metrics, max_post_stress=5.0, min_post_sleep=80.0, min_post_readiness=85.0):
    def objective(sauna_params):
        pred = predict_effects(model, scaler_X, scaler_y, current_metrics, sauna_params)
        post_sleep, post_stress, post_readiness = pred

        penalty = 0
        if post_stress > max_post_stress:
            penalty += 1000 * (post_stress - max_post_stress)
        if post_sleep < min_post_sleep:
            penalty += 1000 * (min_post_sleep - post_sleep)
        if post_readiness < min_post_readiness:
            penalty += 1000 * (min_post_readiness - post_readiness)

        heat_load = sauna_params[0] + sauna_params[1]
        if heat_load > 120:
            penalty += 1000 * (heat_load - 120)

        return -post_readiness + penalty

    temp_range = np.arange(75, 96, 2)
    hum_range = np.arange(10, 36, 2.5)
    dur_range = np.arange(20, 91, 5)

    best_obj = np.inf
    best_params = None

    for temp, hum, dur in product(temp_range, hum_range, dur_range):
        sauna_params = np.array([temp, hum, dur])
        obj = objective(sauna_params)
        if obj < best_obj:
            best_obj = obj
            best_params = sauna_params

    pred = predict_effects(model, scaler_X, scaler_y, current_metrics, best_params)
    return tuple(best_params), pred[0], pred[1], pred[2]

# Main
if __name__ == "__main__":
    model, scaler_X, scaler_y = load_model_and_scalers()

    current_metrics = [75, 7, 65]  # Sleep, Stress, Readiness

    suggested_params, pred_post_sleep, pred_post_stress, pred_post_readiness = tailor_sauna_session(
        model, scaler_X, scaler_y, current_metrics
    )

    recommendation = save_recommendation(
        current_metrics,
        suggested_params,
        pred_post_sleep,
        pred_post_stress,
        pred_post_readiness
    )

    print(f"Temp={recommendation['recommended_temp']}°C, Humidity={recommendation['recommended_hum']}%, Duration={recommendation['recommended_duration']} min")
