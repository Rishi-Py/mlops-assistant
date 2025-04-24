# app_chat.py

import streamlit as st  # ✅ Must come first
st.set_page_config(page_title="MLOps Assistant Chat", layout="wide")

import pandas as pd
from tools import shared_state, experiment_log, log_experiment_func
from assistant import handle_prompt

st.title("🤖 MLOps Assistant (Chat Mode)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1) File Upload & Data Setup
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    shared_state["df"] = df
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    # Chat-based input for target column
    st.markdown("🔍 Enter the **target variable** you want to predict:")
    target = st.text_input("Target variable column name:", key="target_input")
    if target and target in df.columns:
        shared_state["target"] = target
        st.success(f"Target variable set to `{target}`")
    elif target:
        st.warning(f"⚠️ `{target}` not found in dataset columns")

    # Optional: predictor selection
    st.markdown("🧠 (Optional) Enter predictor columns, comma-separated. Leave blank to auto-select:")
    predictor_input = st.text_input("Predictor variables:", key="predictor_input")
    if predictor_input:
        predictors = [col.strip() for col in predictor_input.split(",") if col.strip() in df.columns]
        shared_state["predictors"] = predictors
        st.info(f"Using custom predictors: {', '.join(predictors)}")
    elif "target" in shared_state:
        shared_state["predictors"] = [col for col in df.columns if col != shared_state["target"]]
        st.info(f"Auto-selected predictors (all except target): {', '.join(shared_state['predictors'])}")

# 2) Chat Interface
tab1, tab2 = st.tabs(["🧠 Chat Assistant", "📊 Logs & Downloads"])

with tab1:
    # Render previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Prompt input and assistant response
    if prompt := st.chat_input("Ask me to train, drop columns, monitor, or anything else…"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        response = handle_prompt(prompt)

        # Real-time feedback
        if "✅" in response:
            st.success(response)
        elif "❌" in response:
            st.error(response)
        else:
            st.info(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
        with st.chat_message("assistant"):
            st.markdown(response)

with tab2:
    st.subheader("🧪 Experiment Tracker")

    if st.button("📝 Log This Run"):
        result = log_experiment_func(
            experiment_name="Manual Model Log",
            parameters={"model": "LinearRegression", "note": "Manually logged"},
            result="Simulated RMSE: 4.23"
        )
        st.success(result)

    if experiment_log:
        st.dataframe(pd.DataFrame(experiment_log))
    else:
        st.info("No experiments logged yet. Run a model or log one manually.")

    st.divider()
    st.subheader("📥 Download Processed Dataset")
    if "df" in shared_state:
        st.download_button(
            label="Download Current Dataset",
            data=shared_state["df"].to_csv(index=False).encode("utf-8"),
            file_name="processed_dataset.csv",
            mime="text/csv"
        )
    else:
        st.warning("No dataset available yet. Please upload a CSV.")




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
