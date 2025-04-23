# openai_test.py
import os
from dotenv import load_dotenv
import openai

load_dotenv()  # loads .env

key = os.getenv("OPENAI_API_KEY")
print("KEY LOADED:", key)
openai.api_key = key

try:
    resp = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[{"role":"user","content":"Hello"}]
    )
    print("SUCCESS:", resp.choices[0].message.content)
except Exception as e:
    print("ERROR:", e)
