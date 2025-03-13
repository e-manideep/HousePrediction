import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")


def load_data():
    """
    Loads the California Housing dataset and returns it as a DataFrame.
    """
    housing = fetch_california_housing(as_frame=True)
    df = housing.frame
    return df


def perform_eda(df):
    """
    Performs basic exploratory data analysis:
      - Prints head and statistical description.
      - Checks for missing values.
      - Generates and saves a correlation heatmap.
    """
    print("Dataset Head:")
    print(df.head())
    print("\nDataset Description:")
    print(df.describe())
    print("\nMissing Values:")
    print(df.isnull().sum())

    # Plot and save the correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("correlation_heatmap.png")
    plt.close()


def preprocess_data(df):
    """
    Separates features and target, applies scaling using StandardScaler.
    Although the California dataset has no missing values,
    a placeholder for imputation is included if needed.
    """
    # The target variable is 'MedHouseVal'
    X = df.drop("MedHouseVal", axis=1)
    y = df["MedHouseVal"]

    # Initialize and fit the scaler on the feature set
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def train_and_optimize(X_train, y_train):
    """
    Trains a Random Forest Regressor using GridSearchCV for hyperparameter tuning.
    """
    # Initialize the model
    rf = RandomForestRegressor(random_state=42)

    # Define hyperparameter grid for tuning
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5]
    }

    grid_search = GridSearchCV(estimator=rf,
                               param_grid=param_grid,
                               cv=5,
                               scoring="neg_mean_squared_error",
                               n_jobs=-1,
                               verbose=2)
    grid_search.fit(X_train, y_train)
    print("Best Hyperparameters:", grid_search.best_params_)
    best_model = grid_search.best_estimator_
    return best_model


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the model on the test set and prints RMSE, MAE, and R² scores.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    print(f"Evaluation Metrics:\nRMSE: {rmse:.4f}\nMAE: {mae:.4f}\nR²: {r2:.4f}")
    return rmse, mae, r2


def main():
    # Load the dataset
    df = load_data()

    # Perform exploratory data analysis
    perform_eda(df)

    # Preprocess the data (scaling, splitting features and target)
    X_scaled, y, scaler = preprocess_data(df)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Train and optimize the model using GridSearchCV
    best_model = train_and_optimize(X_train, y_train)

    # Evaluate the optimized model
    evaluate_model(best_model, X_test, y_test)

    # Save the trained model and scaler for deployment
    joblib.dump(best_model, "best_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("Model and scaler saved successfully.")


if __name__ == "__main__":
    main()