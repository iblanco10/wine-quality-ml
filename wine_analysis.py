import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

# Load the datasets
red_wine = pd.read_csv("winequality-red.csv", sep=";")
white_wine = pd.read_csv("winequality-white.csv", sep=";")

# Add a 'type' column: 0 for red, 1 for white
red_wine["type"] = 0
white_wine["type"] = 1

# Combine datasets
data = pd.concat([red_wine, white_wine], axis=0)

# Select features and target
X = data[["fixed acidity", "volatile acidity", "citric acid", "residual sugar",
          "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density",
          "pH", "sulphates", "alcohol"]]  # Only 11 features
y = data["quality"]  # Target variable (Wine Quality)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a simple Linear Regression model
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# Save model and scaler
joblib.dump(model, "wine_quality_model.pkl")
joblib.dump(scaler, "scaler.pkl")
