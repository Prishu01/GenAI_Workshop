import json
import os
import requests
from dotenv import load_dotenv
from tools import get_current_weather
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _groq_chat_completions(model, messages, tools=None, tool_choice=None):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment")

    payload = {
        "model": model,
        "messages": messages,
    }
    if tools is not None:
        payload["tools"] = tools
    if tool_choice is not None:
        payload["tool_choice"] = tool_choice

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    response = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers=headers,
        json=payload,
    )
    response.raise_for_status()
    return response.json()


#important parts of the tool definition 
#weather tool is a json object 
weather_tool = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Get current weather of any city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City Name"
                }
            },
            "required": ["location"]
        }
    }
}

query = input("Ask:")

response = _groq_chat_completions(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": query
        }
    ],
    tools=[weather_tool],
    tool_choice="auto",
)
message = response["choices"][0]["message"]

if message.get("tool_calls"):

    print("\nLLM decided to call a tool...\n")

    tool_call = message["tool_calls"][0]

    function_name = tool_call["function"]["name"]

    arguments = json.loads(tool_call["function"]["arguments"])

    if function_name == "get_current_weather":
        tool_result = get_current_weather(arguments["location"])

    final_response = _groq_chat_completions(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful AI assistant."
            },
            {
                "role": "user",
                "content": query
            },
            message,
            {
                "role": "tool",
                "tool_call_id": tool_call["id"],
                "content": json.dumps(tool_result)
            }
        ]
    )

    print("\nFinal Answer:\n")
    print(final_response["choices"][0]["message"]["content"])

else:
    print("\nLLM did not call any tools:\n")
    print(message["content"])