
from dotenv import load_dotenv
from langsmith import traceable
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.tracers.langchain import wait_for_all_tracers

load_dotenv()


# =============================================
# 1. DEMO STORE DATA
# =============================================

ORDERS = {
    "ORD-1001": "Shipped; expected in 2 business days.",
    "ORD-1002": "Processing; not yet shipped.",
    "ORD-1003": "Delivered successfully.",
    "ORD-1004": "Delayed; customer service should investigate.",
}

STOCK = {
    "wireless mouse": 12,
    "mechanical keyboard": 0,
    "usb-c cable": 24,
    "laptop stand": 6,
}


# =============================================
# 2. CREATE TOOLS
# =============================================

@tool
def check_order_status(order_id: str) -> str:
    """Look up an order's shipping status by order ID."""

    order_id = order_id.strip().upper()
    status = ORDERS.get(order_id)

    if status is None:
        return f"Order {order_id} was not found."

    return f"Order {order_id}: {status}"


@tool
def check_product_stock(product_name: str) -> str:
    """Check the available quantity of a product."""

    name = product_name.strip().lower()

    product = next(
        (key for key in STOCK if name == key or name in key),
        None
    )

    if product is None:
        return f"Product '{product_name}' was not found."

    quantity = STOCK[product]

    if quantity == 0:
        return f"{product.title()} is out of stock."

    return f"{product.title()} has {quantity} units available."


@tool
def get_return_policy() -> str:
    """Get the store's return and refund policy."""

    return (
        "Products can be returned within 30 days of delivery. "
        "Products must be unused and in original packaging. "
        "Faulty items require human support review. "
        "Refunds require approval from customer support."
    )


@tool
def get_shipping_policy() -> str:
    """Get available shipping methods, costs and delivery times."""

    return (
        "Standard delivery: 3-5 business days, €4.99. "
        "Express delivery: 1-2 business days, €12.99. "
        "Free standard shipping for orders of €50 or more."
    )


# =============================================
# 3. CREATE LANGCHAIN AGENT
# =============================================

agent = create_agent(
    model="openai:gpt-4.1-mini",

    tools=[
        check_order_status,
        check_product_stock,
        get_return_policy,
        get_shipping_policy
    ],

    system_prompt=(
        "You are DemoShop's helpful customer support assistant. "
        "Use the appropriate tools to answer questions about "
        "orders, product stock, returns and shipping. "
        "Always use relevant tools before answering these questions. "
        "Never invent order information, prices or policies. "
        "If an order ID is missing, ask the customer for it. "
        "You cannot process refunds, place orders or change accounts. "
        "Be concise, professional and helpful."
    )
)


# =============================================
# 4. CREATE TEST QUESTIONS
# =============================================

DEMO_QUESTIONS = [

    # Order Tracking
    "Where is order ORD-1001?",
    "Has my order ORD-1002 shipped yet?",
    "Was ORD-1003 delivered?",
    "My package ORD-1004 seems late. What happened?",
    "Can you track order ORD-9999?",

    # Inventory
    "Is the wireless mouse in stock?",
    "Can I buy a mechanical keyboard right now?",
    "Do you have any USB-C cables?",
    "How many laptop stands are available?",
    "Do you sell desk lamps?",

    # Returns
    "What is your return policy?",
    "Can I return an unused product after 10 days?",
    "My product arrived broken. What should I do?",
    "Do I need approval to get a refund?",
    "Can I return a product after 45 days?",

    # Shipping
    "How much is normal delivery?",
    "What is the fastest shipping option?",
    "When is shipping free?",
    "How many days does standard shipping take?",
    "Does express delivery cost extra?",

    # Combined Questions
    "Is the wireless mouse available and what does delivery cost?",
    "ORD-1001 is shipped, but how long does standard delivery normally take?",
    "Is the keyboard in stock, and what is your return policy?",
    "Can I return an item and get it shipped back by express delivery?",
    "Do you have laptop stands and offer free shipping?",

    # Unsupported Requests
    "Can you change my account password?",
    "Can you issue my refund right now?",
    "Can you order a pizza for me?",
    "Can you update my billing card information?",
    "Where is my order?"
]


# =============================================
# 5. CUSTOM LANGSMITH TRACING
# =============================================

@traceable(
    name="DemoShop_Customer_Request",
    run_type="chain",
    tags=["demoshop", "customer-support"]
)
def handle_customer_question(question: str) -> str:
    """
    Process a customer question and create
    a custom LangSmith trace.
    """

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": question
            }
        ]
    })

    answer = response["messages"][-1].content

    return str(answer)


# =============================================
# 6. EXECUTION SETTINGS
# =============================================

# False = Run one question
# True  = Run all 30 questions

RUN_DEMO_BATCH = False

SINGLE_QUESTION = "Where is order ORD-1001?"


# =============================================
# 7. MAIN FUNCTION
# =============================================

def main():

    if RUN_DEMO_BATCH:
        questions = DEMO_QUESTIONS
    else:
        questions = [SINGLE_QUESTION]

    print(f"Running {len(questions)} questions...")

    try:

        for index, question in enumerate(questions, start=1):

            print(f"\nQuestion {index}: {question}")

            # Call our traceable function
            answer = handle_customer_question(question)

            print(f"Agent Answer: {answer}")

    finally:

        # Complete pending LangChain tracing operations
        wait_for_all_tracers()

    print("\nExecution completed!")
    print("Check LangSmith for recorded traces.")


# =============================================
# 8. RUN PROGRAM
# =============================================

if __name__ == "__main__":
    main()
