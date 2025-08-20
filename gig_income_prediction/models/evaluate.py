import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import mean_squared_error, r2_score
from utils.io_utils import load_data

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return {
        'mse': mse,
        'r2': r2,
        'predictions': y_pred
    }

def plot_feature_importances(model, feature_names):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), np.array(feature_names)[indices], rotation=90)
    plt.xlim([-1, len(importances)])
    plt.tight_layout()
    plt.show()

def generate_performance_report(model, X_test, y_test, feature_names):
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Mean Squared Error: {metrics['mse']}")
    print(f"R^2 Score: {metrics['r2']}")
    
    plot_feature_importances(model, feature_names)

def main(model_path, X_test_path, y_test_path):
    model = joblib.load(model_path)
    X_test = load_data(X_test_path)
    y_test = load_data(y_test_path)

    feature_names = X_test.columns.tolist()
    generate_performance_report(model, X_test, y_test, feature_names)

if __name__ == "__main__":
    # Example usage
    # main('path/to/model.pkl', 'path/to/X_test.csv', 'path/to/y_test.csv')
    pass  # This file is intentionally left blank for direct execution.