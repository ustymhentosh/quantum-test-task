import pandas as pd
import joblib
import os

def main():
    print("Loading model and test data...")
    if not os.path.exists('model/model.joblib'):
        print("Error: 'model/model.joblib' not found. Please run train.py first.")
        return
    
    if not os.path.exists('data/hidden_test.csv'):
        print("Error: 'data/hidden_test.csv' not found. Please ensure it is in the data/ directory.")
        return

    # Load artifacts
    model = joblib.load('model/model.joblib')
    test_df = pd.read_csv('data/hidden_test.csv')

    print("Engineering features on test data...")
    # Isolate and transform only the necessary features
    X_test = pd.DataFrame()
    X_test['6_squared'] = test_df['6'] ** 2
    X_test['7'] = test_df['7']

    print("Generating predictions...")
    predictions = model.predict(X_test)

    # Create submission dataframe
    submission = pd.DataFrame({
        'prediction': predictions
    })

    # Save to CSV
    output_filename = 'data/predictions.csv'
    submission.to_csv(output_filename, index=False)
    print(f"Predictions successfully saved to '{output_filename}'")

if __name__ == "__main__":
    main()