import requests
from dotenv import load_dotenv
import os


load_dotenv()
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")

if not WEATHER_API_KEY:
    raise ValueError("WEATHER_API_KEY not found. Check your .env file.")

def get_current_weather(location: str):
    url = "https://api.weatherapi.com/v1/current.json" 
    
    #parameter or argument to be passed in the API request
    params = {
        "key": WEATHER_API_KEY,
        "q": location
    }

    response = requests.get(url, params=params)
    response.raise_for_status()  # Raise an error for bad responses
    
    data = response.json()
    
    current = data["current"]

    return{
        "location": data["location"]["name"],
        "region": data["location"]["region"],
        "country": data["location"]["country"],
        "temperature_c": current["temp_c"],
        "condition": current["condition"]["text"],
        "humidity": current["humidity"],
        "wind_kph": current["wind_kph"]
    }