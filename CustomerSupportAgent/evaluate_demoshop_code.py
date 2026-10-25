"""Code-based LangSmith evaluation for the DemoShop customer-support agent.

Run: python evaluate_demoshop_code.py
Uses the existing 'demoshop-support-dataset' and tests whether the
agent calls exactly the expected tool(s) in each dataset example's metadata.
"""

from dotenv import load_dotenv
from langsmith import evaluate

from customer_support_agent import agent

load_dotenv()

DATASET_NAME = "demoshop-support-dataset"


# 1. Run the agent on one question from the dataset.
def run_agent(inputs: dict) -> dict:
    result = agent.invoke({
        "messages": [{"role": "user", "content": inputs["question"]}]
    })

    # LangChain AI messages record the tools the model called.
    tools_used = [
        call["name"]
        for message in result["messages"]
        for call in (getattr(message, "tool_calls", None) or [])
    ]

    return {
        "answer": str(result["messages"][-1].content),
        "tools_used": tools_used,
    }


# 2. Code-based evaluator: no extra LLM judge is needed.
def check_tool_selection(run, example) -> dict:
    expected = set((example.metadata or {}).get("expected_tools", []))
    actual = set((run.outputs or {}).get("tools_used", []))

    return {
        "key": "correct_tool_selection",
        "score": int(actual == expected),
        "comment": f"Expected: {sorted(expected)}; Used: {sorted(actual)}",
    }


# 3. Run the agent and evaluator across the existing dataset.
if __name__ == "__main__":
    evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[check_tool_selection],
        experiment_prefix="demoshop-code-evaluation",
        max_concurrency=2,
    )

    print("Evaluation complete. Open LangSmith > Datasets & Experiments.")
