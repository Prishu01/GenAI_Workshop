import os
import pandas as pd
import matplotlib.pyplot as plt
import smtplib
from email.message import EmailMessage
from dotenv import load_dotenv

from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq

# ----------------------------
# Load Environment Variables
# ----------------------------
load_dotenv()

llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

CSV_FILE = "data/Bank.csv"

# -------------------------------------------------
# Tool 1 : Analyze Expenses
# -------------------------------------------------
def analyze_expenses(dummy=""):
    """Analyze income, expenses and monthly savings."""

    df = pd.read_csv(CSV_FILE)

    income = df[df["Type"] == "Credit"]["Amount"].sum()
    expense = df[df["Type"] == "Debit"]["Amount"].sum()

    savings = income - expense

    return f"""
Financial Summary

Total Income : ₹{income:,.2f}

Total Expenses : ₹{expense:,.2f}

Monthly Savings : ₹{savings:,.2f}
"""


# -------------------------------------------------
# Tool 2 : Spending Categories
# -------------------------------------------------
def spending_categories(dummy=""):
    """Show spending by category."""

    df = pd.read_csv(CSV_FILE)

    category = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    return category.to_string()


# -------------------------------------------------
# Tool 3 : Highest Spending
# -------------------------------------------------
def highest_spending(dummy=""):
    """Find the highest spending category."""

    df = pd.read_csv(CSV_FILE)

    category = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest = category.idxmax()
    amount = category.max()

    return f"""
You are spending the most on

{highest}

Amount : ₹{amount:,.2f}

Top Categories

{category.to_string()}
"""


# -------------------------------------------------
# Tool 4 : Budget Suggestions
# -------------------------------------------------
def budget_suggestions(dummy=""):
    """Provide financial suggestions."""

    df = pd.read_csv(CSV_FILE)

    category = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest = category.idxmax()

    suggestions = []

    suggestions.append(f"Highest Spending Category : {highest}\n")

    if highest == "Shopping":
        suggestions.append(
            "Reduce online shopping and create a monthly shopping budget."
        )

    elif highest == "Food":
        suggestions.append(
            "Reduce food delivery and cook at home."
        )

    elif highest == "Travel":
        suggestions.append(
            "Plan trips in advance to reduce travel expenses."
        )

    elif highest == "Rent":
        suggestions.append(
            "Rent is consuming a large part of your income."
        )

    suggestions.append(
        "Save at least 20% of your monthly income."
    )

    suggestions.append(
        "Maintain an emergency fund."
    )

    suggestions.append(
        "Track discretionary expenses every week."
    )

    return "\n".join(suggestions)


# -------------------------------------------------
# Tool 5 : Expense Chart
# -------------------------------------------------
def create_chart(dummy=""):
    """Create expense chart."""

    df = pd.read_csv(CSV_FILE)

    category = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
    )

    plt.figure(figsize=(10, 5))

    category.plot(kind="bar")

    plt.title("Expense Category Analysis")

    plt.xlabel("Category")

    plt.ylabel("Amount")

    plt.tight_layout()

    plt.savefig("expense_chart.png")

    plt.close()

    return "Expense chart saved as expense_chart.png"


# -------------------------------------------------
# Tool 6 : Email Report
# -------------------------------------------------
def email_report(dummy=""):
    """Send financial report via email."""

    df = pd.read_csv(CSV_FILE)

    income = df[df["Type"] == "Credit"]["Amount"].sum()
    expense = df[df["Type"] == "Debit"]["Amount"].sum()
    savings = income - expense

    category = (
        df[df["Type"] == "Debit"]
        .groupby("Category")["Amount"]
        .sum()
        .sort_values(ascending=False)
    )

    highest = category.idxmax()
    highest_amount = category.max()

    report = f"""
AI FINANCIAL ADVISOR REPORT
===========================

Total Income     : ₹{income:,.2f}

Total Expenses   : ₹{expense:,.2f}

Monthly Savings  : ₹{savings:,.2f}

Highest Spending : {highest}

Amount           : ₹{highest_amount:,.2f}

------------------------------------
Category Wise Spending
------------------------------------

{category.to_string()}

------------------------------------
Budget Suggestions
------------------------------------

1. Reduce spending in the highest category.

2. Save at least 20% of your monthly income.

3. Track weekly expenses.

4. Build an emergency fund.

5. Continue investing wisely.

Generated using AI Financial Advisor.
"""

    sender = os.getenv("EMAIL_USER")
    password = os.getenv("EMAIL_PASSWORD")
    receiver = "priyankavk8878@gmail.com"

    if not sender or not password or not receiver:
        return "Email credentials are missing in the .env file."

    msg = EmailMessage()

    msg["Subject"] = "Monthly Financial Report"

    msg["From"] = sender

    msg["To"] = receiver

    msg.set_content(report)

    # Attach expense chart if it exists
    if os.path.exists("expense_chart.png"):
        with open("expense_chart.png", "rb") as f:
            msg.add_attachment(
                f.read(),
                maintype="image",
                subtype="png",
                filename="expense_chart.png"
            )

    try:

        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:

            smtp.starttls()

            smtp.login(sender, password)

            smtp.send_message(msg)

        return f"Financial report emailed successfully to {receiver}"

    except Exception as e:
        return f"Email failed : {e}"


# -------------------------------------------------
# Tools
# -------------------------------------------------
tools = [

    Tool(
        name="Expense Analysis",
        func=analyze_expenses,
        description="Analyze income, expenses and monthly savings."
    ),

    Tool(
        name="Spending Categories",
        func=spending_categories,
        description="Shows spending category wise."
    ),

    Tool(
        name="Highest Spending",
        func=highest_spending,
        description="Finds highest expense category."
    ),

    Tool(
        name="Budget Suggestions",
        func=budget_suggestions,
        description="Provides financial suggestions."
    ),

    Tool(
        name="Expense Chart",
        func=create_chart,
        description="Creates expense bar chart."
    ),
    Tool(
    name="Email Report",
    func=email_report,
    description="Emails the complete financial report with expense chart."
    ),

]


# -------------------------------------------------
# Memory
# -------------------------------------------------
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# -------------------------------------------------
# Agent
# -------------------------------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="zero-shot-react-description",
    verbose=True
)

print("\n==============================")
print(" AI Financial Advisor Started ")
print("==============================")

print("\nType 'exit' to quit.\n")

# -------------------------------------------------
# Chat Loop
# -------------------------------------------------
while True:

    query = input("You : ")

    if query.lower() == "exit":
        break

    response = agent.invoke(
        {
            "input": query
        }
    )

    print("\nAdvisor :")
    print(response["output"])