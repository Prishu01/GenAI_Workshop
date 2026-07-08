from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv(override =True)

api_key= os.getenv("GROQ_API_KEY")

client= OpenAI(
     api_key=api_key,
     base_url="https://api.groq.com/openai/v1"
)

topic =input("Enter the topic for the email: ")

prompt = f"""
Write a professional email about:

{topic}
"""

response= client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

print("\nGenerated Email:\n")
print(response.choices[0].message.content)