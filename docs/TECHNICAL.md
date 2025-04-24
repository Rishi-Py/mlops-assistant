# MLOps Assistant – Technical Overview

## 🧱 System Architecture

  Prompt → Streamlit App → Prompt Handler → Tools (training, deploy, logs) → Output

## ⚙️ Code Structure

- `app_chat.py` – Streamlit UI with rule-based logic
- `app_agent.py` – Streamlit UI powered by LangChain agent
- `assistant.py` – Handles rule-based prompts
- `assistant_agent.py` – Routes prompts to LangChain agent tools
- `tools.py` – All model training, cleaning, deployment functions
- `config.py` – Loads OpenAI API key from `.env`
- `datasets/` – Uploaded and processed CSVs
- `experiment_logs/` – Stores experiment history

## 🧪 Tools Implemented

- Train model
- Fix missing data
- Drop text columns
- Monitor performance
- Deploy to staging/production
- Provision GPU
- Log experiment
- Get/update model version

## 🔧 Tech Stack

- Python
- Streamlit
- scikit-learn
- LangChain + OpenAI
- dotenv
- GitHub (for version control)

## 🛣️ What’s Next

- Add real deployment (e.g. FastAPI, Docker)
- Integrate MLflow for versioning
- Replace simulated infra with real cloud (AWS, GCP)


