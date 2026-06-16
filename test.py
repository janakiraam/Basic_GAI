# from langchain_ollama import ChatOllama
# from datetime import datetime
#
# llm = ChatOllama(
#     model="gemma4"
# )
#
# start_time = datetime.now()
# print("Start Time: \n", start_time)
#
# response = llm.invoke("How can I use gemma4 for free? Not running in my local")
# print(response.content)
#
# end_time = datetime.now()
# print("\n \n \n End Time:", end_time)
#
# print("\n \n \n Execution Time:", end_time - start_time)


# --- Groq version ---
# Requires: pip install langchain-groq python-dotenv
# Put GROQ_API_KEY=... in the .env file located beside this script.

import os
from datetime import datetime
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables from .env in the same folder as this script
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

"""
- `llama-3.3-70b-versatile`
- `llama-3.1-8b-instant`
- `meta-llama/llama-4-scout-17b-16e-instruct`
- `meta-llama/llama-4-maverick-17b-128e-instruct`
- `qwen/qwen3-32b`
- `deepseek-r1-distill-llama-70b`

"""

llm = ChatGroq(
    #model="llama-3.3-70b-versatile",   # pick any supported Groq model
    model="openai/gpt-oss-120b",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY"),
)

start_time = datetime.now()
print("Start Time: \n", start_time)

response = llm.invoke("How can I use Groq for free? Not running in my local")
print(response.content)

end_time = datetime.now()
print("\n \n \n End Time:", end_time)

print("\n \n \n Execution Time:", end_time - start_time)
