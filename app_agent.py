# app_chat.py

import streamlit as st  # ✅ Streamlit must come FIRST
st.set_page_config(page_title="MLOps Assistant Chat", layout="wide")  # ✅ Must be FIRST Streamlit command

# ✅ Only now import other libraries
import pandas as pd
from tools import shared_state
from assistant_agent import handle_prompt

st.title("🤖 MLOps Assistant (Intelligent Chat Mode)")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1) File upload & state
uploaded_file = st.file_uploader("Upload your CSV dataset", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())
    target = st.selectbox("Select target column", df.columns)
    shared_state["df"] = df
    shared_state["target"] = target

# Show download button if data is loaded
if "df" in shared_state:
    st.download_button(
        label="📥 Download Processed Dataset",
        data=shared_state["df"].to_csv(index=False).encode("utf-8"),
        file_name="processed_dataset.csv",
        mime="text/csv"
    )

# 2) Render past messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 3) Chat input & response
if prompt := st.chat_input("Ask me to train, drop columns, monitor, or anything else…"):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate assistant reply
    response = handle_prompt(prompt)

    # Optional feedback to the user
    if "✅" in response:
        st.success(response)
    elif "❌" in response:
        st.error(response)
    else:
        st.info(response)

    # Store and display assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)


from tools import experiment_log

if experiment_log:
    st.sidebar.header("🧪 Experiment Tracker")
    st.sidebar.dataframe(pd.DataFrame(experiment_log))


tab1, tab2 = st.tabs(["🧠 Chat Assistant", "📊 Logs & Downloads"])

with tab1:
    # Everything related to chat UI
    ...  # Keep your existing chat history + prompt code here

with tab2:
    st.subheader("🧪 Experiment Tracker")

    from tools import experiment_log, log_experiment_func

    # Manual log button
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
