"""Create or update a LangSmith evaluation dataset for the DemoShop agent.

Run: python create_demoshop_dataset.py

- Creates 'demoshop-support-dataset' if missing.
- If it already exists, updates matching questions and adds new questions.
- Preserves the dataset ID, existing experiments, and unrelated examples.
- Does not run the agent, call an LLM, or produce traces.

Matches the fictional store data and DEMO_QUESTIONS in
'demoshop_support_agent_traceable.py'.
"""

from collections import Counter
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

DATASET_NAME = "demoshop-support-dataset"

# Each entry: (customer question, reference answer / key facts, expected tools).
# Reference answers are short statements of the required facts, NOT exact strings
# the LLM must reproduce. Use an LLM-as-Judge for semantic correctness.
CASES = {
    "order_tracking": [
        ("Where is order ORD-1001?", "ORD-1001 has shipped; expected in 2 business days.", ["check_order_status"]),
        ("Has my order ORD-1002 shipped yet?", "ORD-1002 is processing and has not shipped yet.", ["check_order_status"]),
        ("Was ORD-1003 delivered?", "ORD-1003 was delivered successfully.", ["check_order_status"]),
        ("My package ORD-1004 seems late. What happened?", "ORD-1004 is delayed; customer service should investigate.", ["check_order_status"]),
        ("Can you track order ORD-9999?", "ORD-9999 was not found in the demo system; do not invent a status.", ["check_order_status"]),
    ],
    "inventory": [
        ("Is the wireless mouse in stock?", "Wireless mouse is in stock with 12 units available.", ["check_product_stock"]),
        ("Can I buy a mechanical keyboard right now?", "Mechanical keyboard is out of stock (0 units available).", ["check_product_stock"]),
        ("Do you have any USB-C cables?", "USB-C cable is in stock with 24 units available.", ["check_product_stock"]),
        ("How many laptop stands are available?", "There are 6 laptop stands available.", ["check_product_stock"]),
        ("Do you sell desk lamps?", "Desk lamps are not found in the demo catalog; avoid inventing availability.", ["check_product_stock"]),
    ],
    "returns": [
        ("What is your return policy?", "Returns are allowed within 30 days of delivery if unused and in original packaging; support approval is required.", ["get_return_policy"]),
        ("Can I return an unused product after 10 days?", "A return after 10 days is within the 30-day window, subject to original packaging and support approval.", ["get_return_policy"]),
        ("My product arrived broken. What should I do?", "A faulty item requires human support review; do not promise an automatic refund.", ["get_return_policy"]),
        ("Do I need approval to get a refund?", "Yes, a customer support representative must approve returns and refunds.", ["get_return_policy"]),
        ("Can I return a product after 45 days?", "45 days is outside the standard 30-day return window; do not promise eligibility.", ["get_return_policy"]),
    ],
    "shipping": [
        ("How much is normal delivery?", "Standard delivery costs €4.99; free standard delivery applies on orders of €50 or more.", ["get_shipping_policy"]),
        ("What is the fastest shipping option?", "Express delivery is the fastest listed option: 1–2 business days for €12.99.", ["get_shipping_policy"]),
        ("When is shipping free?", "Standard delivery is free for orders of €50 or more.", ["get_shipping_policy"]),
        ("How many days does standard shipping take?", "Standard delivery takes 3–5 business days.", ["get_shipping_policy"]),
        ("Does express delivery cost extra?", "Express delivery costs €12.99 and takes 1–2 business days.", ["get_shipping_policy"]),
    ],
    "combined": [
        ("Is the wireless mouse available and what does delivery cost?", "Wireless mouse: 12 in stock. Standard shipping €4.99 (free from €50); express €12.99.", ["check_product_stock", "get_shipping_policy"]),
        ("ORD-1001 is shipped, but how long does standard delivery normally take?", "ORD-1001 shipped and is expected in 2 business days; standard delivery normally takes 3–5 business days.", ["check_order_status", "get_shipping_policy"]),
        ("Is the keyboard in stock, and what is your return policy?", "Mechanical keyboard out of stock; standard returns within 30 days of delivery if unused and in original packaging with support approval.", ["check_product_stock", "get_return_policy"]),
        ("Can I return an item and get it shipped back by express delivery?", "Returns require support approval within 30 days if unused and packaged. Express outbound delivery costs €12.99 for 1–2 business days; return-shipping arrangements are not specified.", ["get_return_policy", "get_shipping_policy"]),
        ("Do you have laptop stands and offer free shipping?", "Laptop stands: 6 in stock. Free standard shipping on orders of €50 or more.", ["check_product_stock", "get_shipping_policy"]),
    ],
    "unsupported_or_missing_info": [
        ("Can you change my account password?", "The agent cannot change account passwords; explain this limitation.", []),
        ("Can you issue my refund right now?", "The agent cannot issue refunds; a support representative must approve refunds.", ["get_return_policy"]),
        ("Can you order a pizza for me?", "Ordering pizza is outside the agent's customer-support scope.", []),
        ("Can you update my billing card information?", "The agent cannot modify payment or account details.", []),
        ("Where is my order?", "Ask the customer for their order ID before checking status; do not invent an order.", []),
    ],
}


def build_examples() -> list[dict]:
    """Build dataset examples with question, reference answer, and metadata."""
    examples = []
    for category, cases in CASES.items():
        for question, answer, expected_tools in cases:
            examples.append({
                "inputs": {"question": question},
                "outputs": {"answer": answer},
                "metadata": {
                    "category": category,
                    "expected_tools": expected_tools,
                    "source": "DemoShop fictional test cases",
                },
            })
    assert len(examples) == 30, "Expected 30 DemoShop examples"
    assert len({e["inputs"]["question"] for e in examples}) == 30, "Duplicate questions found"
    return examples


def main() -> None:
    client = Client()
    examples = build_examples()

    existing_datasets = list(client.list_datasets(dataset_name=DATASET_NAME))
    if existing_datasets:
        dataset = next((d for d in existing_datasets if d.name == DATASET_NAME), None)
        if dataset is None:
            raise RuntimeError("Dataset name filter returned an unexpected dataset")
        print(f"Dataset already exists: {dataset.name}")
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description="30 DemoShop customer-support test cases across six categories",
        )
        print(f"Created new dataset: {dataset.name}")

    existing_by_question = {
        e.inputs.get("question"): e
        for e in client.list_examples(dataset_id=dataset.id)
        if e.inputs and e.inputs.get("question")
    }

    new_examples = []
    updated = 0
    for example in examples:
        question = example["inputs"]["question"]
        old = existing_by_question.get(question)
        if old is None:
            new_examples.append(example)
        elif old.outputs != example["outputs"] or (
            (old.metadata or {}).get("category") != example["metadata"]["category"]
            or (old.metadata or {}).get("expected_tools") != example["metadata"]["expected_tools"]
        ):
            # Preserve the existing ID and any other metadata it already has.
            client.update_example(
                example_id=old.id,
                inputs=example["inputs"],
                outputs=example["outputs"],
                metadata={**(old.metadata or {}), **example["metadata"]},
            )
            updated += 1

    if new_examples:
        client.create_examples(dataset_id=dataset.id, examples=new_examples)

    counts = Counter(e["metadata"]["category"] for e in examples)
    print(f"Dataset ID: {dataset.id}")
    print(f"Created {len(new_examples)} new examples; updated {updated}; unchanged {30-len(new_examples)-updated}.")
    print("Categories:", dict(counts))
    print("Find it in LangSmith -> Datasets & Experiments ->", DATASET_NAME)


if __name__ == "__main__":
    main()
