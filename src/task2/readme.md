# Tabular Data Regression:

This repository contains an end-to-end Machine Learning pipeline to predict a continuous target variable from an anonymized tabular dataset. 

While the dataset provides 53 features, extensive Exploratory Data Analysis (EDA) revealed that this is a generated synthetic dataset. Complex, heavy models (like XGBoost or Random Forests) are unnecessary. Instead, the `target` is a mathematical combination of two variables: **Feature 6** and **Feature 7**.

## **Project Guidance & EDA Insights**

Before running the modeling scripts, it is highly recommended to review `eda.ipynb` to understand the data geometry:
1. **Noise:** Features 0-5 and 8-52 are uniformly distributed random variables with small predictive power.
2. **Parabola:** Plotting Feature `6` against the target reveals a shape of parabola. Feature `6` thus is squared in our model to linearize its relationship with the target.
3. **Linear Component:** Once the parabolic effect of Feature `6` is subtracted from the target, the remaining residual is linearly correlated with Feature `7`.

**Conclusion:** By engineering a `6_squared` feature and pairing it with Feature `7` inside a standard `LinearRegression` model, we achieve a near-perfect validation RMSE of $\approx 1.22 \times 10^{-13}$ (essentially $0.0$, accounting for floating-point limits). 

---

## **Project Setup**

Clone the repository and navigate to the project directory:**

```bash
git clone <repository_url>
cd <repository_name>
Create and activate a virtual environment (Recommended):
```

macOS/Linux:

```Bash
python -m venv venv
source venv/bin/activate
```
Windows:

```DOS
python -m venv venv
venv\Scripts\activate
```
Install the required dependencies:

```Bash
pip install -r requirements.txt
```

Ensure that raw data files (train.csv and hidden_test.csv) are placed in the `data/` directory of this project before executing the scripts.

## **Execution Guide**
1. Exploratory Data Analysis (EDA)
To view the visualizations and mathematics behind the model choice - open eda.ipynb and run the cells to see the feature distributions and residual scatter plots.

2. Train the Model
To train the Linear Regression model and save it as an artifact (`model/model.joblib`), execute:

```Bash
python train.py
```
Note: This script will print the validation RMSE to the terminal and automatically retrain on the full train.csv dataset for maximum inference accuracy.

3. Generate Predictions
To load the trained model, automatically apply the same feature engineering to the test data, and generate the final predictions:

```Bash
python predict.py
```
This will output a `data/predictions.csv` file containing the predictions for `data/hidden_test.csv`.