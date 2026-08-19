import argparse
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

def main():
    print("Loading training data...")
    try:
        train_df = pd.read_csv('data/train.csv')
    except FileNotFoundError:
        print("Error: 'data/train.csv' not found. Please ensure it is in the data/ directory.")
        return

    print("Engineering features (Feature 6 squared & Feature 7)...")
    # Extract only the features discovered during EDA
    X = pd.DataFrame()
    X['6_squared'] = train_df['6'] ** 2
    X['7'] = train_df['7']
    y = train_df['target']

    # Split into train and validation sets to verify our ~0.0 RMSE
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    val_preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_preds))
    print(f"Validation RMSE: {rmse}")

    # Retrain on the full dataset to capture all possible data points before inference
    print("Retraining model on full dataset for maximum test accuracy...")
    model.fit(X, y)

    # Save the model
    joblib.dump(model, 'data/model.joblib')
    print("Model successfully saved to 'model.joblib'")

if __name__ == "__main__":
    main()