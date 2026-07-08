import json
import re
import requests
from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_BASE_URL = "https://api.groq.com/openai/v1"


def _groq_chat_completions(model, messages, response_format=None):
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set in environment")

    payload = {
        "model": model,
        "messages": messages,
    }
    if response_format is not None:
        payload["response_format"] = response_format

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


class Response:
    def __init__(self, output):
        self.final_output = output


class Runner:
    @staticmethod
    def run_sync(agent, query):

        # Default prompt
        prompt = ""

        # If the query is about weather, call the tool first
        if "weather" in query.lower():
            location = input("enter location :")
            

            match = re.search(r"in\s+([A-Za-z]+)", query)
            if match:
                location = match.group(1).strip()

            tool = agent.tools[0]
            tool_result = tool(location)

            prompt = f"""
{agent.instructions}

Weather Data:
{json.dumps(tool_result, indent=2)}

Return ONLY JSON in this format:
{{
    "location": "",
    "temperature_c": 0,
    "condition": "",
    "humidity": 0,
    "wind_kph": 0
}}
"""

        else:
            prompt = f"""
{agent.instructions}

Question:
{query}
"""

        # Call the LLM for BOTH weather and normal queries
        response = _groq_chat_completions(
            model=agent.model,
            messages=[
                {
                    "role": "system",
                    "content": prompt
                }
            ],
            response_format={"type": "json_object"}
        )

        result = response["choices"][0]["message"]["content"]

        # Convert to dataclass model if provided
        if agent.output_type:
            if isinstance(result, str):
                try:
                    result = json.loads(result)
                except json.JSONDecodeError:
                    pass
            if isinstance(result, dict):
                result = agent.output_type(**result)

        return Response(result)