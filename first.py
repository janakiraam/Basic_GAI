# hello_genai.py -- Your first Generative AI API call
# Purpose : Send a prompt to Groq's hosted LLM and print the model's reply.
# Setup   : pip install groq python-dotenv langchain-groq
# API key : Get a free key at https://console.groq.com and put it in a .env
#           file next to this script as:  GROQ_API_KEY=gsk_xxxxxxxxxxxx

# ---------- 1. Imports ----------
from groq import Groq                       # Official Groq SDK -- used to call the chat completions API
import os                                   # Standard lib -- used to read environment variables and build paths
from datetime import datetime               # Imported (not used here) -- handy if you later want to time the call
from dotenv import load_dotenv              # Reads key=value pairs from a .env file into environment variables
from langchain_groq import ChatGroq         # Imported (not used in this file) -- LangChain wrapper around Groq

# ---------- 2. Load secrets from .env ----------
# os.path.dirname(__file__)  -> folder where THIS .py file lives
# os.path.join(..., ".env")  -> full path to the .env file in that same folder
# load_dotenv(path)          -> reads that file and injects its variables into os.environ
load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))

# ---------- 3. Create the Groq client ----------
# os.getenv("GROQ_API_KEY") fetches the key that load_dotenv just loaded.
# Groq(api_key=...) builds an authenticated client we can use to call the API.
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ---------- 4. Send a chat completion request ----------
# This is the actual API call. It takes a list of messages (the conversation)
# plus generation settings, and returns the model's response.
response = client.chat.completions.create(
    model    = "llama-3.3-70b-versatile",   # Which LLM to use on Groq's servers (replaces decommissioned llama3-70b-8192)
    messages = [
        # "system" message -> sets the assistant's persona / behaviour rules
        {"role": "system",  "content": "You are a helpful teacher who explains things simply."},
        # "user" message   -> the actual question we want answered
        {"role": "user",    "content": "can you please explain the code which is running now?."}
    ],
    temperature = 0.7,    # Randomness: 0 = deterministic/repeatable, 1+ = more creative/varied
    max_tokens  = 1000,   # Upper limit on how many tokens (~words/word-pieces) the model may generate
)

# ---------- 5. Print the model's answer ----------
# response.choices is a list of possible replies (usually just one).
# .message.content is the actual text the model produced.
print(response.choices[0].message.content)

# ---------- 6. Show token usage (useful for cost / quota tracking) ----------
# response.usage holds counts of tokens used in this call.
usage = response.usage
# prompt_tokens     = tokens consumed by your input (system + user messages)
# completion_tokens = tokens the model generated as the reply
print(f"Tokens: {usage.prompt_tokens} prompt + {usage.completion_tokens} generated")
