
from typing import Literal

from dotenv import load_dotenv
from langsmith import evaluate
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Reuse existing agent and tools
from evaluate_demoshop_code import run_agent
from customer_support_agent import (
    check_order_status,
    check_product_stock,
    get_return_policy,
    get_shipping_policy
)

load_dotenv()

DATASET = "demoshop-support-dataset"


# ==========================================
# 1. CREATE SECOND AGENT
# ==========================================

detailed_agent = create_agent(
    model="openai:gpt-4.1-mini",
    tools=[
        check_order_status,
        check_product_stock,
        get_return_policy,
        get_shipping_policy
    ],
    system_prompt=(
        "You are DemoShop's customer support assistant. "
        "Always use relevant tools to answer store questions. "
        "Never invent information. "
        "Provide clear, detailed answers with useful next steps. "
        "Ask for an order ID when missing. "
        "You cannot issue refunds or modify accounts."
    )
)


def run_detailed_agent(inputs: dict) -> dict:

    result = detailed_agent.invoke({
        "messages": [
            {"role": "user", "content": inputs["question"]}
        ]
    })

    return {"answer": str(result["messages"][-1].content)}


# ==========================================
# 2. CREATE LLM JUDGE
# ==========================================

class Decision(BaseModel):
    winner: Literal["A", "B", "tie"]
    reason: str = Field(description="Brief explanation")


judge = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).with_structured_output(Decision)


# ==========================================
# 3. EXTRACT ANSWER
# ==========================================

def extract_answer(data):

    if isinstance(data, dict):
        for key in ("answer", "output", "outputs"):
            if key in data:
                return extract_answer(data[key])

    return str(data if data is not None else "")


# ==========================================
# 4. PAIRWISE EVALUATOR
# ==========================================

def pairwise_judge(runs, example) -> dict:

    answer_a = extract_answer(runs[0].outputs)
    answer_b = extract_answer(runs[1].outputs)

    question = example.inputs["question"]
    expected = extract_answer(example.outputs)

    decision = judge.invoke([
        (
            "system",
            "Compare two customer support answers for "
            "correctness, helpfulness and clarity. "
            "Use the reference answer for facts. "
            "Return tie when both are equally good."
        ),
        (
            "human",
            f"Question: {question}\n"
            f"Reference: {expected}\n"
            f"Answer A: {answer_a}\n"
            f"Answer B: {answer_b}"
        )
    ])

    if decision.winner == "A":
        scores = [1.0, 0.0]
    elif decision.winner == "B":
        scores = [0.0, 1.0]
    else:
        scores = [0.5, 0.5]

    return {
        "key": "pairwise_answer_quality",
        "scores": {
            str(runs[0].id): scores[0],
            str(runs[1].id): scores[1]
        },
        "comment": decision.reason
    }


# ==========================================
# 5. RUN PAIRWISE EVALUATION
# ==========================================

if __name__ == "__main__":

    # Original DemoShop agent
    original = evaluate(
        run_agent,
        data=DATASET,
        experiment_prefix="demoshop-original",
        max_concurrency=2
    )

    # Detailed DemoShop agent
    detailed = evaluate(
        run_detailed_agent,
        data=DATASET,
        experiment_prefix="demoshop-detailed",
        max_concurrency=2
    )

    # Compare both experiments
    evaluate(
        (
            original.experiment_name,
            detailed.experiment_name
        ),
        evaluators=[pairwise_judge],
        randomize_order=True,
        experiment_prefix="demoshop-pairwise",
        max_concurrency=2
    )

    print("Pairwise evaluation completed!")
