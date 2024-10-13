import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse
import mlflow
import mlflow.sklearn
import requests
import io
import logging
import dagshub

# Set up basic logging
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Function to evaluate metrics for model performance
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    # Read the wine-quality CSV file from the URL
    csv_url = "https://raw.githubusercontent.com/shrikant-temburwar/Wine-Quality-Dataset/refs/heads/master/winequality-red.csv"
    
    try:
        # Step 1: Download the CSV file from the URL
        response = requests.get(csv_url)
        if response.status_code == 200:
            logger.info("CSV file downloaded successfully")

            # Step 2: Read the CSV content into pandas
            data = pd.read_csv(io.StringIO(response.text), sep=";")
            logger.info("CSV file read successfully")
        else:
            logger.error(f"Failed to download file. Status code: {response.status_code}")
    except Exception as e:
        logger.exception("Unable to download or read CSV, check your internet connection. Error: %s", e)
        sys.exit(1)  # Exit if CSV download fails

    # Proceed only if data is successfully loaded
    if 'data' in locals():
        # Split the data into training and test sets (0.75, 0.25 split)
        train, test = train_test_split(data, test_size=0.25, random_state=42)
        logger.info("Data split into train and test sets")

        # The predicted column is "quality" which is a scalar from [3, 9]
        train_x = train.drop(["quality"], axis=1)
        test_x = test.drop(["quality"], axis=1)
        train_y = train[["quality"]]
        test_y = test[["quality"]]

        # Parameters for ElasticNet model (default values if not provided via CLI)
        alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
        l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

        # Start the MLflow run
        with mlflow.start_run():
            lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
            lr.fit(train_x, train_y)

            # Predict the test set
            predicted_qualities = lr.predict(test_x)

            # Evaluate the metrics
            (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

            print(f"ElasticNet model (alpha={alpha}, l1_ratio={l1_ratio}):")
            print(f"  RMSE: {rmse}")
            print(f"  MAE: {mae}")
            print(f"  R2: {r2}")

            # Log parameters and metrics to MLflow
            mlflow.log_param("alpha", alpha)
            mlflow.log_param("l1_ratio", l1_ratio)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)

            # Log the model
            remote_server_url = "https://dagshub.com/Lavishgangwani/MLflow-test.mlflow"
            mlflow.set_tracking_uri(remote_server_url)
    
            dagshub.init(repo_owner='Lavishgangwani', repo_name='MLflow-test', mlflow=True)
 
            tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

            # Model registry only works with non-file stores
            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(lr, "model", registered_model_name="ElasticNetWineModel")
            else:
                mlflow.sklearn.log_model(lr, "model")