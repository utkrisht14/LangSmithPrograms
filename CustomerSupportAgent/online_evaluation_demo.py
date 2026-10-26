
from dotenv import load_dotenv
from langchain_core.tracers.langchain import wait_for_all_tracers

load_dotenv()

from customer_support_agent import handle_customer_question


questions = [
    "Where is order ORD-1001?",
    "Is the wireless mouse available?",
    "Can you issue my refund right now?",
    "What is the express shipping cost?"
]

try:
    for question in questions:
        answer = handle_customer_question(question)

        print("\nCustomer:", question)
        print("Agent:", answer)

finally:
    wait_for_all_tracers()
