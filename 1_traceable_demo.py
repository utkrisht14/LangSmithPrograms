import os
from langsmith import traceable
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGSMITH_TRACING"] = "true"

@traceable
def double_number(n):
    return n * 2

print(double_number(4))
