import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

client = Client()

print("Configured project:", os.getenv("LANGSMITH_PROJECT", "default"))
print("\nAvailable LangSmith projects:")

for project in client.list_projects(limit=100):
    print("-", project.name)