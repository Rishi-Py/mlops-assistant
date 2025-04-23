# assistant_agent.py

from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from config import OPENAI_API_KEY
from tools import (
    train_model_from_state_func,
    drop_non_numeric_columns_func,
    monitor_model_func,
    cross_validation_func,  # Import the new function
    fix_missing_data_func,
    drop_specified_columns_func,
    check_data_types_func,
    get_model_version_func,
    update_model_version_func,
    deploy_model_func,
    log_experiment_func,
    hyperparameter_tuning_func,
    feature_selection_func,
    trigger_deployment_pipeline_func,
    provision_gpu_func,
    shared_state
)


# Initialize OpenAI LLM
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

# Define tools for LangChain Agent
tools = [
    Tool(
        name="Train Model",
        func=train_model_from_state_func,
        description="Train a regression or classification model using the current dataset and selected target column."
    ),
    Tool(
        name="Drop Non-Numeric Columns",
        func=drop_non_numeric_columns_func,
        description="Remove all non-numeric columns from the dataset."
    ),
    Tool(
        name="Fix Missing Data",
        func=fix_missing_data_func,
        description="Fill missing values (NaNs) in the dataset using column means."
    ),
    Tool(
        name="Drop Specified Columns",
        func=lambda prompt: drop_specified_columns_func(columns=prompt.split(",")),
        description="Drop specific columns by name. Provide a comma-separated list of column names."
    ),
    Tool(
        name="Check Data Types",
        func=check_data_types_func,
        description="Check whether the target column is numeric or categorical."
    ),
    Tool(
        name="Monitor Model",
        func=monitor_model_func,
        description="Simulate monitoring model drift or performance over time."
    ),
    Tool(
        name="Get Model Version",
        func=get_model_version_func,
        description="Get the current version of the trained model."
    ),
    Tool(
        name="Update Model Version",
        func=update_model_version_func,
        description="Update the model version to a new timestamp-based version."
    ),
    Tool(
        name="Deploy Model",
        func=lambda env: deploy_model_func(env),
        description="Simulate deployment to 'staging' or 'production'. Provide the environment."
    ),
    Tool(
        name="Log Experiment",
        func=lambda _: log_experiment_func(
            experiment_name="Agent Logged Experiment",
            parameters={"model": "LinearRegression"},
            result="Simulated RMSE: 4.23"
        ),
        description="Log an experiment to the experiment tracker."
    ),
]

# Initialize agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type="zero-shot-react-description",
    verbose=True
)

# Unified handler
def handle_prompt(prompt: str) -> str:
    try:
        return agent.run(prompt)
    except Exception as e:
        return f"❌ Agent error: {e}"


Tool(
    name="Cross-Validation",
    func=cross_validation_func,
    description="Perform K-fold cross-validation to evaluate model stability."
)


Tool(
    name="Hyperparameter Tuning",
    func=hyperparameter_tuning_func,
    description="Tuning model hyperparameters using Grid Search (with cross-validation)."
)


Tool(
    name="Feature Selection",
    func=feature_selection_func,
    description="Perform feature selection to keep the most statistically significant features."
)

Tool(
    name="Deploy Model",
    func=deploy_model_func,
    description="Simulate deployment to staging or production."
)


Tool(
    name="Trigger Deployment Pipeline",
    func=trigger_deployment_pipeline_func,
    description="Simulate triggering a deployment pipeline (e.g., GitHub Actions)."
)


Tool(
    name="Provision GPU",
    func=provision_gpu_func,
    description="Simulate GPU provisioning for model training."
)



