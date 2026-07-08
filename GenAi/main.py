from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

from agent import Agent
from runner import Runner
from tools import get_current_weather_data

@dataclass
class Weather:
    location: str
    temperature_c: float
    condition: str
    humidity: float
    wind_kph: float

weather_agent = Agent(
    name="Weather Agent",
    instructions= """
You are an expert weather assistant. You have access to a tool that provides current weather information for a given location. When a user asks about the weather, you should use the tool to fetch the data and return it in a structured JSON format. If the query does not pertain to weather
""",
tools=[get_current_weather_data],
model="llama-3.3-70b-versatile",
output_type=Weather
)
response = Runner.run_sync(
    weather_agent,
    "What is the current weather in {location}?")

print(response.final_output)


