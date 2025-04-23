# app.py

import streamlit as st
import pandas as pd
from assistant import handle_prompt

st.set_page_config(page_title="MLOps Assistant", layout="wide")
st.title("🧠 MLOps Assistant")

prompt = st.text_input("Enter your assistant prompt:")
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df.head())

    target = st.selectbox("Select target column", df.columns)

    if prompt and target:
        st.write(f"📝 Prompt: {prompt}")
        result, model = handle_prompt(prompt, df, target)
        st.success(result)
