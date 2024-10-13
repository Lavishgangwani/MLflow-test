# MLFLOW-TEST

This project is an example of using **MLflow** to track machine learning experiments. The repository demonstrates how to set up and run an ML experiment with ElasticNet regression, using **DagsHub** for remote MLflow tracking.

## Setup

To track experiments on **DagsHub**, you need to set up the MLflow tracking URI along with your credentials (username and password). Below are the steps to run the project.

### Step 1: Set Up Environment Variables

Make sure you export the required environment variables for tracking on DagsHub. Use the following commands:

```bash
export MLFLOW_TRACKING_URI=https://dagshub.com/Lavishgangwani/MLFLOW-TEST.mlflow
export MLFLOW_TRACKING_USERNAME=LavishGangwani
export MLFLOW_TRACKING_PASSWORD=<your_dagshub_token>
```

### Step 2: Run the Python Script

Once the environment variables are set, you can run the `script.py` to start training and tracking the machine learning model:

```bash
python example.py
```

### Project Overview

- The project uses **ElasticNet** regression to train a model on the Wine Quality dataset.
- The script logs the following metrics to MLflow:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared value (R²)

You can view the logged parameters and metrics by visiting the MLflow UI hosted on DagsHub.

### MLflow Tracking

This project tracks and logs:
    - Model parameters: `alpha` and `l1_ratio`
    - Performance metrics: `rmse`, `mae`, `r2`
    - The trained model

To view the experiment and model details, navigate to your **MLflow tracking server** on DagsHub:

[MLFLOW-TEST Tracking UI](https://dagshub.com/Lavishgangwani/MLflow-test.mlflow)

### Important Notes

- Ensure you keep your **MLFLOW_TRACKING_PASSWORD** secure and do not share it publicly.


### License

This project is licensed under the MIT License - see the LICENSE file for details.

---