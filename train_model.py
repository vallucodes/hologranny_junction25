import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import sqlite3
import pickle

# Set seed for reproducibility
np.random.seed(43)
torch.manual_seed(43)

# Load data from database
def load_data_from_database(db_path='database.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM sauna_sessions", conn)
    conn.close()
    print(f"Loaded {len(df)} records from database")
    return df

# Dataset class
class PostSaunaDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

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

# Save function
def save_model_and_scalers(model, scaler_X, scaler_y, model_path='model.pth', scaler_X_path='scaler_X.pkl', scaler_y_path='scaler_y.pkl'):
    torch.save(model.state_dict(), model_path)
    with open(scaler_X_path, 'wb') as f:
        pickle.dump(scaler_X, f)
    with open(scaler_y_path, 'wb') as f:
        pickle.dump(scaler_y, f)
    print(f"Saved model to {model_path} and scalers to {scaler_X_path}, {scaler_y_path}")

# Train function
def train_model(df, num_epochs=500, batch_size=16):
    features = ['pre_sleep', 'pre_stress', 'pre_readiness', 'avg_temp', 'avg_hum', 'duration']
    targets = ['post_sleep', 'post_stress', 'post_readiness']
    X = df[features].values
    y = df[targets].values
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)
    train_dataset = PostSaunaDataset(X_train, y_train)
    test_dataset = PostSaunaDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = PostSaunaMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    print("\nTraining model...")
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss/len(train_loader):.4f}")
    # Evaluate
    print("\nEvaluating model...")
    model.eval()
    y_pred_scaled = []
    y_true_scaled = []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            y_pred_scaled.append(outputs.numpy())
            y_true_scaled.append(y_batch.numpy())
    y_pred_scaled = np.vstack(y_pred_scaled)
    y_true_scaled = np.vstack(y_true_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_true = scaler_y.inverse_transform(y_true_scaled)
    mse_sleep = mean_squared_error(y_true[:, 0], y_pred[:, 0])
    mse_stress = mean_squared_error(y_true[:, 1], y_pred[:, 1])
    mse_readiness = mean_squared_error(y_true[:, 2], y_pred[:, 2])
    print(f"\nTest MSE - Sleep: {mse_sleep:.2f}, Stress: {mse_stress:.2f}, Readiness: {mse_readiness:.2f}")
    return model, scaler_X, scaler_y

# Main: Run training
if __name__ == "__main__":
    df = load_data_from_database()
    model, scaler_X, scaler_y = train_model(df)
    save_model_and_scalers(model, scaler_X, scaler_y)
