#!/usr/bin/env python
# coding: utf-8


# In[44]:


import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt


# In[45]:


import sys
import os

# Calculate the project root (three levels up from this notebook)
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)


# In[46]:


from src.house_price_predictor.model.estimators import ScikitLearnEstimator


# In[47]:


# Set MLflow tracking URI (adjust port if needed)
mlflow.set_tracking_uri(uri="http://localhost:8080")
mlflow.set_experiment("house-price-prediction")


# In[48]:


# Load processed train and test data
train_df = pd.read_parquet("../../../data/staged/train.parquet")
test_df = pd.read_parquet("../../../data/staged/test.parquet")
train_df["price"] = train_df["price"].astype("float64")
test_df["price"] = test_df["price"].astype("float64")


# In[ ]:


# Separate features and target
target = "price"
X_train = train_df.drop(columns=[target])
y_train = train_df[target]
X_test = test_df.drop(columns=[target])
y_test: pd.Series = test_df[target]

models = [
    LinearRegression(),
    RandomForestRegressor(random_state=42),
    XGBRegressor(random_state=42)
]
X_train


# In[ ]:


def plot_stuff(X_test, y_test, model, estimator, y_pred):
    plt.figure(figsize=(8,4))
    plt.scatter(x=X_test["area"], y=y_test, label="Actual", alpha=0.5)
    plt.scatter(x=X_test["area"], y=y_pred, label="Predicted", alpha=0.5)
        
        #plt.hist(y_pred, bins=50, alpha=0.5, label="Predicted", color="blue")
        #plt.hist(y_test, bins=50, alpha=0.5, label="Actual", color="orange")
    print()
    plt.legend()
    plt.title(f"Scatterplot of {estimator.estimator_name} predictions")
    plt.show()
        # Calculate prediction errors
    errors = y_test - y_pred
        # Create and log error box plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.boxplot(errors)
    ax.set_title(f"Prediction Error Distribution for {model.__class__.__name__}")
    ax.set_ylabel("Error (Actual Price - Predicted Price)")
    ax.set_xticklabels([model.__class__.__name__]) # Label the box with model name
    plt.show(fig)
    return errors

for model in models:
    with mlflow.start_run(run_name=model.__class__.__name__):
        estimator = ScikitLearnEstimator(model)
        estimator.train(X_train, y_train)
        y_pred: pd.Series = estimator.predict(X_test)
        metrics = estimator.evaluate(y_test, y_pred)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model",)
        print(f"\nLogged metrics for {estimator.estimator_name}")
        for metric_name, value in metrics.items():
            print(f"{metric_name}: {value:.4f}")
        print()
        errors = plot_stuff(X_test, y_test, model, estimator, y_pred)
        # Create a DataFrame for detailed error analysis
        error_details_df = pd.DataFrame({
            'Actual_Price': y_test,
            'Predicted_Price': y_pred,
            'Error': errors,
            'Absolute_Error': errors.abs(),
            'Percentage_Error': (errors / y_test).abs() * 100
        })

        # Sort by absolute error in descending order and get top 10
        top_10_biggest_errors = error_details_df.sort_values(by='Absolute_Error', ascending=False).head(10)

        print(f"\n--- Top 10 Biggest Prediction Errors for {model.__class__.__name__} (by absolute value) ---")
        print(top_10_biggest_errors.to_string()) # .to_string() ensures the full DataFrame is printed
        print("-" * 70) # Separator line


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np

# Assuming you have already fit your model, e.g.:
# rf = RandomForestRegressor().fit(X_train, y_train)
# or
# xgb = XGBRegressor().fit(X_train, y_train)

def plot_feature_importances(model, feature_names, top_n=10):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]
    plt.figure(figsize=(8,4))
    plt.title("Feature Importances")
    plt.bar(range(top_n), importances[indices], align="center")
    plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45)
    plt.show()

# Example usage:
plot_feature_importances(models[1], X_train.columns)
plot_feature_importances(models[2], X_train.columns)


# In[ ]:




