from dotenv import load_dotenv
load_dotenv(override=True)
import os
import requests
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

@tool
def download_sales_data(dummy: str = "") -> str:
    """
    Download sales CSV file.
    """
    url = "https://drive.google.com/uc?export=download&id=1x4UDtoONLJnA_MT093FaeBaLpntYaXPl"
    response = requests.get(url)

    with open("sales_data.csv", "wb") as f:
        f.write(response.content)
    return "sales_data.csv downloaded successfully"

@tool
def analyze_sales_data(file_path: str) -> str:
    """
    read csv and analyze sales data
    """
    df = pd.read_csv(file_path)
    summary = df.describe().to_string()
    return summary

@tool
def create_sales_chart(file_path: str) -> str:
    """
    create sales chart visualization
    """
    df = pd.read_csv(file_path)
    numeric_cols = df.select_dtypes(include="number").columns

    if len(numeric_cols) == 0:
        return "No numeric column found"
    column = numeric_cols[0]

    plt.figure(figsize=(8, 5))
    plt.plot(df[column])
    plt.title(f"{column} Trend")
    plt.xlabel("index")
    plt.ylabel(column)
    plt.savefig("sales_chart.png")
    plt.close()

    return "chart saved as sales_chart.png"

@tool
def email_report(dummy: str = "") -> str:
    """
    Email sales chart report
    """
    receiver_email = "rishupandey99936@gmail.com"
    sender_email = os.getenv("EMAIL_USER")
    sender_password = os.getenv("EMAIL_PASSWORD")
    msg = EmailMessage()

    msg["Subject"] = "AI generated sales report"
    msg["From"] = sender_email
    msg["To"] = receiver_email

    msg.set_content(
        "please find attached AI generated sales report chart."
    )

    with open("sales_chart.png", "rb") as f:
        file_data = f.read()
        msg.add_attachment(
            file_data,
            maintype="image",
            subtype="png",
            filename="sales_chart.png"
        )
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(sender_email, sender_password)
        smtp.send_message(msg)
    return "Email report sent successfully."

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
)

tools = [
    download_sales_data,
    analyze_sales_data,
    create_sales_chart,
    email_report
]

# Correct construction of the chat prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
You are an Autonomous Workflow AI agent.

You can:
- download files
- Analyze CSV data
- Create charts
- Send Email reports
- Execute taks step-by-step autonomously

Always use the available tools whenever required.
"""
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True
)

print("\nAdvanced LangChain + Groq Agent Started")
print("Type 'exit' to stop\n")

while True:
    user_input = input("You : ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye")
        break

    response = agent_executor.invoke(
        {
            "input": user_input
        }
    )

    print("\nBot Response:")
    print("----------------------------------------------")
    print(response["output"])
    print("----------------------------------------------\n")