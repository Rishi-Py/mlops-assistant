# config.py
import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(f"DEBUG: OPENAI_API_KEY = {OPENAI_API_KEY[:6]}*****")


