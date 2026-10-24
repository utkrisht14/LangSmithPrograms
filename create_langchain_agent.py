
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

# Step 1: Create a tool
@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


# Step 2: Create the agent
agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[add_numbers],
    system_prompt=(
        "You are a math assistant. "
        "Always use the add_numbers tool for addition. "
        "Return only the numeric answer."
    )
)


# Step 3: Ask the agent a question
if __name__ == "__main__":
    response = agent.invoke({
        "messages": [
            {"role": "user", "content": "What is 5 + 3?"}
        ]
    })

    print(response["messages"][-1].content)
