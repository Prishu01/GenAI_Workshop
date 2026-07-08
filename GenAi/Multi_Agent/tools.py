import requests
import os
from dotenv import load_dotenv

load_dotenv()

Weather_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_current_weather_data(location:str):
    url="https://api.weatherapi.com/v1/current.json"
    params={
        "key": Weather_API_KEY,
        "q": location
    }

    response=requests.get(url,params=params)
    response.raise_for_status()

    data = response.json()

    current = data['current']

    return {
        "location": data['location']['name'],
        "temperature": current['temp_c'],
        "condition": current['condition']['text'],
        "humidity": current['humidity'],
        "wind": current['wind_kph']
    }