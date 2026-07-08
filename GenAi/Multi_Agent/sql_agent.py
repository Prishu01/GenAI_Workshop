from agent import Agent

sql_agent = Agent(
    name="SQL Agent",
    instructions="""
You are an expert SQL assistant. You have access to a database and can help with writing,
 debugging, and optimizing SQL queries.
"""
)