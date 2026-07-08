import re

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1"
)


class Response:

    def __init__(self, output):
        self.final_output = output


class Runner:

    @staticmethod
    def run_sync(agent, query):

        # Weather Agent
        if agent.name == "Weather Agent":

            location = "London"

            match = re.search(r"in\s+([A-Za-z ]+)", query)

            if match:
                location = match.group(1)

            weather = agent.tools[0](location)

            prompt = f"""
{agent.instructions}

Weather Data

{weather}

Answer the user.
"""

        else:

            prompt = f"""
{agent.instructions}

Question

{query}
"""

        response = client.chat.completions.create(

            model=agent.model,

            messages=[
                {
                    "role":"system",
                    "content":prompt
                }
            ]
        )

        return Response(
            response.choices[0].message.content
        )