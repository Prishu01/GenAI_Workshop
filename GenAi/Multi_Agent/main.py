from food_agent import food_agent
from sql_agent import sql_agent
from Pcb_agent import pcb_agent
from weather_agent import weather_agent
from coding_agent import coding_agent
from runner import Runner

query = input("Ask Anything : ")

# Supervisor Logic

weather_keywords = [
    "weather",
    "temperature",
    "rain",
    "humidity",
    "forecast"
]

coding_keywords = [
    "python",
    "java",
    "oop",
    "exception",
    "api",
    "program",                                       
    "class"
]

Sql_keywords = [
    "sql",
    "database",
    "query",
    "table",
    "insert",
    "delete",
    "update"
]

Pcb_keywords=[
    "pcb",
    "circuit",
    "board",
    "layout",
    "component",
    "trace",
    "solder",
    "schematic"
]

food_keywords=[
    "recipe",
    "cooking",
    "ingredients",
    "dish",
    "meal",
    "cuisine",
    "chef"
    "desert",
    "baking",
    "grilling",
    "frying"
]

if any(word in query.lower() for word in weather_keywords):

    selected_agent = weather_agent

elif any(word in query.lower() for word in coding_keywords):

    selected_agent = coding_agent

elif any(word in query.lower() for word in Sql_keywords):

    selected_agent = sql_agent
elif any(word in query.lower() for word in Pcb_keywords):

    selected_agent = pcb_agent
elif any(word in query.lower() for word in food_keywords):

    selected_agent = food_agent

else:

    print("No suitable agent found.")
    exit()

print(f"\nDelegating to : {selected_agent.name}\n")

response = Runner.run_sync(
    selected_agent,
    query
)

print(response.final_output)