import datetime
import os

import pandas as pd
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_groq import ChatGroq

# ----------------------------------------------------
# Load Environment Variables
# ----------------------------------------------------
load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY not found!\n"
        "Create a .env file and add:\n\n"
        "GROQ_API_KEY=your_groq_api_key"
    )

# ----------------------------------------------------
# Tool 1 : Current Date & Time
# ----------------------------------------------------
@tool
def get_current_time(dummy: str = "") -> str:
    """Returns the current date and time."""
    return str(datetime.datetime.now())


# ----------------------------------------------------
# Tool 2 : Calculator
# ----------------------------------------------------
@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""

    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"


# ----------------------------------------------------
# Tool 3 : CSV Analyzer
# ----------------------------------------------------
@tool
def analyse_csv(file_path: str) -> str:
    """
    Reads a CSV file and returns the average
    of all numeric columns.
    """

    try:
        df = pd.read_csv(file_path)

        numeric = df.mean(numeric_only=True)

        if numeric.empty:
            return "No numeric columns found."

        return numeric.to_string()

    except Exception as e:
        return f"Error reading CSV: {e}"


# ----------------------------------------------------
# Memory
# ----------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# ----------------------------------------------------
# LLM
# ----------------------------------------------------
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=api_key
)

# ----------------------------------------------------
# Tools
# ----------------------------------------------------
tools = [
    get_current_time,
    calculator,
    analyse_csv
]

# ----------------------------------------------------
# Prompt
# ----------------------------------------------------
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant with tool calling abilities."
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        (
            "human",
            "{input}"
        ),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# ----------------------------------------------------
# Agent
# ----------------------------------------------------
agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ----------------------------------------------------
# Executor
# ----------------------------------------------------
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

# ----------------------------------------------------
# Chat Loop
# ----------------------------------------------------
print("\n==============================")
print(" LangChain + Groq Agent Ready ")
print("==============================")
print("Type 'exit' to quit.\n")

while True:

    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Bot: Goodbye!")
        break

    try:
        response = agent_executor.invoke(
            {
                "input": user_input
            }
        )

        print("\nBot:")
        print("--------------------------------")
        print(response["output"])
        print("--------------------------------\n")

    except Exception as e:
        print(f"\nError: {e}\n")