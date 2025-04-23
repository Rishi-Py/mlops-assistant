# assistant.py

from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY
from tools import (
    shared_state,
    train_model_from_state_func,
    drop_non_numeric_columns_func,
    monitor_model_func,
    fix_missing_data_func,
    drop_specified_columns_func,
    check_data_types_func,
    get_model_version_func,
    update_model_version_func,
    deploy_model_func,
    log_experiment_func
)

# Initialize LLM for fallback use only
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

def handle_prompt(prompt: str) -> str:
    t = prompt.lower()

    # --- DROP tools ---
    if "drop" in t and ("text" in t or "non numeric" in t or "non-numeric" in t):
        return drop_non_numeric_columns_func()

    if "drop columns" in t:
        try:
            columns = [col.strip() for col in t.split("drop columns")[1].split(",")]
            return drop_specified_columns_func(columns=columns)
        except IndexError:
            return "❌ Please specify column names after 'drop columns'."

    # --- CLEANING / DEBUG ---
    if "fix missing data" in t:
        return fix_missing_data_func()
    if "check data types" in t:
        return check_data_types_func()

    # --- MODEL TRAINING & MONITORING ---
    if "train" in t:
        return train_model_from_state_func()
    if "monitor" in t:
        return monitor_model_func()

    # --- VERSIONING / DEPLOYMENT / LOGGING ---
    if "get model version" in t:
        return get_model_version_func()
    if "update model version" in t:
        return update_model_version_func()
    if "deploy" in t and "model" in t:
        try:
            environment = t.split("deploy model to")[1].strip()
            return deploy_model_func(environment)
        except IndexError:
            return "❌ Please specify environment: 'staging' or 'production'."
    if "log experiment" in t:
        return log_experiment_func(
            experiment_name="Model Training Experiment",
            parameters={"model": "LinearRegression", "test_size": 0.2},
            result="Trained with RMSE: 4.23"
        )

    # --- LLM fallback for general questions ---
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"❌ LLM fallback error: {e}"

    # 2) LLM fallback for undefined tasks
    try:
        response = llm.invoke(prompt)
        return response
    except Exception as e:
        return f"❌ LLM fallback error: {e}"
