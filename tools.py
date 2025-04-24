# tools.py

import pandas as pd
import datetime
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score
from langchain.tools import Tool

# Shared global state to hold uploaded DataFrame and target column
shared_state = {}

# Global variable to track the model version
model_version = "1.0"  # You can manually update this or use a date-based versioning system

# --- Tool for Training Models ---
def train_model_from_state_func(*args, **kwargs) -> str:
    """
    Trains a regression or classification model based on shared_state['df'] and ['target'].
    """
    df = shared_state.get("df")
    target = shared_state.get("target")

    if df is None or target is None:
        return "❌ Missing data or target. Please upload a CSV and select a target column."

    try:
        # Drop rows with NaN in the target column
        df_clean = df.dropna(subset=[target])

        X = df_clean.drop(columns=[target]).select_dtypes(include=["number"])
        y = df_clean[target]
        task = "regression" if y.nunique() > 5 else "classification"

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        if task == "regression":
            model = LinearRegression()
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            rmse = mean_squared_error(y_test, preds)  # Returns MSE instead of RMSE
            return f"✅ Trained Linear Regression | RMSE: {rmse:.2f}"

        else:
            model = LogisticRegression(max_iter=200)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            return f"✅ Trained Logistic Regression | Accuracy: {acc:.2f}"

    except Exception as e:
        return f"❌ Training failed: {e}"

# --- Tool for Fixing Missing Data ---
def fix_missing_data_func(*args, **kwargs) -> str:
    """
    Fix missing data (NaN) by filling with the mean of the column.
    This only applies to numerical columns.
    """
    df = shared_state.get("df")
    
    if df is None:
        return "❌ No data loaded. Please upload a CSV first."

    try:
        # Select only numerical columns for filling missing data
        numerical_cols = df.select_dtypes(include=["number"]).columns

        # Fill missing data in numerical columns with the mean
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())
        
        shared_state["df"] = df
        return f"✅ Filled missing values with mean for numerical columns."

    except Exception as e:
        return f"❌ Error fixing missing data: {e}"


# --- Tool for Dropping Columns ---
def drop_specified_columns_func(*args, **kwargs) -> str:
    """
    Drop user-specified columns from the dataset.
    """
    df = shared_state.get("df")

    if df is None:
        return "❌ No data loaded. Please upload a CSV first."

    try:
        # Extract columns to drop
        columns_to_drop = kwargs.get('columns', [])
        df_dropped = df.drop(columns=columns_to_drop, errors="ignore")
        shared_state["df"] = df_dropped
        return f"✅ Dropped columns: {', '.join(columns_to_drop)}."
    except Exception as e:
        return f"❌ Error dropping columns: {e}"

# --- Tool for Checking Data Types ---
def check_data_types_func(*args, **kwargs) -> str:
    """
    Check for incorrect data types in the target column or features.
    """
    df = shared_state.get("df")
    target = shared_state.get("target")

    if df is None or target is None:
        return "❌ Missing data or target column. Please upload a CSV and select the target."

    try:
        # Check if target is numeric for regression, or categorical for classification
        target_type = df[target].dtype

        if target_type == 'object':  # Categorical type (may need encoding)
            return f"⚠️ Target column '{target}' is categorical. Consider encoding it for classification."
        elif target_type in ['float64', 'int64']:  # Numeric type (for regression)
            return f"✅ Target column '{target}' is numeric. Proceeding with regression."

        return f"❌ Unexpected data type in target column. Type: {target_type}"

    except Exception as e:
        return f"❌ Error checking data types: {e}"

# --- Tool for Model Versioning ---
def get_model_version_func(*args, **kwargs) -> str:
    """
    Returns the current model version.
    """
    return f"✅ Current model version: {model_version}"

def update_model_version_func(*args, **kwargs) -> str:
    """
    Updates the model version to the current date.
    """
    global model_version
    model_version = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"✅ Model version updated to: {model_version}"

# --- Tool for Experiment Logging ---
experiment_log = []

def log_experiment_func(experiment_name: str, parameters: dict, result: str, *args, **kwargs) -> str:
    """
    Log experiment details.
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_log.append({
        "timestamp": timestamp,
        "experiment_name": experiment_name,
        "parameters": parameters,
        "result": result
    })
    return f"✅ Experiment '{experiment_name}' logged at {timestamp}."

# --- Tool for Model Deployment ---
deployment_log = []

def deploy_model_func(environment: str, *args, **kwargs) -> str:
    """
    Simulate deploying the model to a specified environment (staging or production).
    Tracks deployment events in a log.
    """
    if environment not in ["staging", "production"]:
        return "❌ Invalid environment. Choose 'staging' or 'production'."
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    deployment_log.append({
        "timestamp": timestamp,
        "environment": environment,
        "status": "success"
    })
    
    return f"✅ Model successfully deployed to {environment} at {timestamp}."

# --- Tool for Model Monitoring ---
model_performance = []

def monitor_model_func(*args, **kwargs) -> str:
    """
    Simulate model drift monitoring and return performance change over time.
    """
    accuracy = 0.98  # Placeholder for actual monitoring metrics
    drift_percentage = 2  # Simulated drift
    model_performance.append({
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "accuracy": accuracy - drift_percentage / 100  # Simulated drift
    })
    
    return f"📊 Simulated drift detected: Accuracy down by {drift_percentage}% over the last 7 days."

# --- Register all tools ---
train_model_from_state = Tool(
    name="Train a model",
    func=train_model_from_state_func,
    description="Train a regression or classification model using the uploaded dataset and selected target column. Returns RMSE or Accuracy."
)

fix_missing_data = Tool(
    name="Fix missing data",
    func=fix_missing_data_func,
    description="Fix missing data by filling NaNs with the mean of the column."
)

drop_specified_columns = Tool(
    name="Drop specified columns",
    func=drop_specified_columns_func,
    description="Drop specified columns from the dataset."
)

check_data_types = Tool(
    name="Check data types",
    func=check_data_types_func,
    description="Check for correct data types in the target column and features."
)

get_model_version = Tool(
    name="Get model version",
    func=get_model_version_func,
    description="Get the current version of the trained model."
)

update_model_version = Tool(
    name="Update model version",
    func=update_model_version_func,
    description="Update the model version to the current date and time."
)

log_experiment = Tool(
    name="Log experiment",
    func=log_experiment_func,
    description="Log an experiment with its parameters and results."
)

deploy_model = Tool(
    name="Deploy model",
    func=deploy_model_func,
    description="Simulate deployment of the model to a given environment (staging or production)."
)

monitor_model = Tool(
    name="Monitor model",
    func=monitor_model_func,
    description="Simulate monitoring the model for drift in performance over time."
)


# tools.py

def drop_non_numeric_columns_func(*args, **kwargs) -> str:
    """
    Drops all non-numeric columns from shared_state['df'].
    """
    df = shared_state.get("df")
    
    if df is None:
        return "❌ No data loaded. Please upload a CSV first."
    
    try:
        # Keep only numeric columns
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        shared_state["df"] = df[numeric_cols]
        return f"✅ Dropped non-numeric columns. Remaining columns: {numeric_cols}"
    except Exception as e:
        return f"❌ Error dropping non-numeric columns: {e}"

#--- K-fold cross validation 

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

def cross_validation_func(*args, **kwargs):
    df = shared_state.get("df")
    target = shared_state.get("target")

    if df is None or target is None:
        return "❌ No data or target selected. Please upload a dataset and select a target column."

    try:
        X = df.drop(columns=[target])
        y = df[target]

        model = RandomForestRegressor()  # You can swap this with any model
        cv_scores = cross_val_score(model, X, y, cv=5)

        return f"✅ Cross-validation scores: {cv_scores} | Mean score: {cv_scores.mean():.2f}"

    except Exception as e:
        return f"❌ Error during cross-validation: {e}"


#--- hyperparameter tuning - grid search

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def hyperparameter_tuning_func(*args, **kwargs):
    df = shared_state.get("df")
    target = shared_state.get("target")

    if df is None or target is None:
        return "❌ No data or target selected. Please upload a dataset and select a target column."

    try:
        X = df.drop(columns=[target])
        y = df[target]

        model = LinearRegression()  # You can swap this with any model
        param_grid = {'fit_intercept': [True, False], 'normalize': [True, False]}

        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X, y)

        best_params = grid_search.best_params_
        return f"✅ Best hyperparameters: {best_params}"

    except Exception as e:
        return f"❌ Error during hyperparameter tuning: {e}"


#--- feature selection 

from sklearn.feature_selection import SelectKBest, f_regression

def feature_selection_func(*args, **kwargs):
    df = shared_state.get("df")
    target = shared_state.get("target")

    if df is None or target is None:
        return "❌ No data or target selected. Please upload a dataset and select a target column."

    try:
        X = df.drop(columns=[target])
        y = df[target]

        selector = SelectKBest(f_regression, k="all")
        selector.fit(X, y)

        selected_features = X.columns[selector.get_support()]
        return f"✅ Selected features based on statistical significance: {', '.join(selected_features)}"

    except Exception as e:
        return f"❌ Error during feature selection: {e}"

#--- model deployment

# Simulate Deployment with status logging
def deploy_model_func(environment: str, *args, **kwargs) -> str:
    """
    Simulate model deployment to staging or production, logging each deployment.
    """
    if environment not in ["staging", "production"]:
        return "❌ Invalid environment. Please choose 'staging' or 'production'."
    
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    deployment_log.append({
        "timestamp": timestamp,
        "environment": environment,
        "status": "success"
    })
    
    # Log the deployment
    return f"✅ Model deployed to {environment} at {timestamp}."


#--- CI/CD simulation 

def trigger_deployment_pipeline_func(*args, **kwargs) -> str:
    """
    Simulate triggering a deployment pipeline (e.g., GitHub Actions).
    """
    pipeline_status = random.choice(["success", "failure"])
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Log pipeline status
    return f"✅ Triggered pipeline at {timestamp} | Status: {pipeline_status}"



#---- GPU provisioning and infra scaling

def provision_gpu_func(*args, **kwargs) -> str:
    """
    Simulate provisioning a GPU for model training.
    """
    # Simulate provisioning process
    gpu_id = f"GPU-{random.randint(1000, 9999)}"
    status = random.choice(["provisioned", "failed"])
    
    if status == "provisioned":
        return f"✅ {gpu_id} provisioned successfully."
    else:
        return f"❌ Failed to provision GPU."



import datetime, random
from tools import shared_state  # assuming shared_state already holds df, predictors, target

# --- Tool: Recommend infra based on data size & model type ---
def recommend_infra_func(*args, **kwargs) -> str:
    df = shared_state.get("df")
    target = shared_state.get("target")
    predictors = shared_state.get("predictors", [])
    if df is None or target is None:
        return "❌ No dataset or target selected. Upload data first."
    n_rows, n_feats = df.shape[0], len(predictors)
    # Simple rules
    if n_rows > 200_000 or n_feats > 50:
        choice = "Databricks cluster"
    elif "GPU" in kwargs.get("requires", "") or df[target].dtype == "float64":
        choice = "Azure ML GPU compute"
    else:
        choice = "Kubernetes CPU node pool"
    return (f"✅ Based on {n_rows} rows and {n_feats} features, I recommend: **{choice}** "
            f"(estimated cost: ${random.randint(50,200)}/day)")

# --- Tool: Provision infra (simulate) ---
infra_log = []
def provision_infra_func(environment: str, *args, **kwargs) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    infra_log.append({"time": now, "env": environment, "action": "provisioned"})
    return f"✅ {environment} provisioned at {now}"

# --- Tool: Scale infra ---
def scale_infra_func(replicas: int = 1, *args, **kwargs) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    infra_log.append({"time": now, "env": "last", "action": f"scaled to {replicas} replicas"})
    return f"✅ Scaled infrastructure to {replicas} replicas at {now}"

# --- Tool: Decommission infra ---
def decommission_infra_func(*args, **kwargs) -> str:
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    infra_log.append({"time": now, "env": "last", "action": "decommissioned"})
    return f"✅ Infrastructure decommissioned at {now}"

# --- Tool: Show infra log ---
def show_infra_log_func(*args, **kwargs) -> str:
    if not infra_log:
        return "ℹ️ No infra actions recorded yet."
    # Format as markdown
    lines = ["| Time | Environment | Action |", "|-----|-------------|--------|"]
    for e in infra_log:
        lines.append(f"| {e['time']} | {e['env']} | {e['action']} |")
    return "\n".join(lines)

# Register (if using in assistant.py) or import these in assistant_agent.py


