import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.xgboost
from urllib.parse import urlparse


# Setup logging
import logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_classification_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred, average='macro')
    recall = recall_score(actual, pred, average='macro')
    f1 = f1_score(actual, pred, average='macro')
    return accuracy, precision, recall, f1


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    
    # Command-line arguments for hyperparameters
    learning_rate = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
    max_depth = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    # Load the BMI dataset
    data = pd.read_csv("bmi.csv")
    
        
    # Encode the 'Gender' column
    le = LabelEncoder()
    data['Gender'] = le.fit_transform(data['Gender'])
    
    # Split the data into training and test sets
    X = data.drop(["Index"], axis=1)
    y = data["Index"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Initialize an XGBoost classifier with command-line specified or default hyperparameters
    model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth, use_label_encoder=False, eval_metric='mlogloss')
    
    with mlflow.start_run():
        # Train the model
        model.fit(X_train, y_train)
        
        # Predict on the test set
        predictions = model.predict(X_test)
        
        # Evaluate the model
        accuracy, precision, recall, f1 = eval_classification_metrics(y_test, predictions)
        
        # Log parameters, metrics, and the model to MLflow
        mlflow.log_params({"learning_rate": learning_rate, "max_depth": max_depth})
        mlflow.log_metrics({"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1})
        
        # Log the model
        mlflow.xgboost.log_model(model, "model")
        
        print(f"Model performance:\n- Accuracy: {accuracy}\n- Precision: {precision}\n- Recall: {recall}\n- F1 Score: {f1}")

        # DAGsHub MLflow Tracking URI
        remote_server_uri = "https://dagshub.com/Abhi0323/BMI-Prediction-with-MLflow-DagsHub.mlflow"
        mlflow.set_tracking_uri(remote_server_uri)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        
        # Check if the tracking URI scheme is not 'file' for model registration
        if tracking_url_type_store != "file":
            mlflow.xgboost.log_model(
                model, "model", registered_model_name="BMI-XGBoost-Model"
            )
        else:
            mlflow.xgboost.log_model(model, "model")
        
        print(f"Model performance:\n- Accuracy: {accuracy}\n- Precision: {precision}\n- Recall: {recall}\n- F1 Score: {f1}")
