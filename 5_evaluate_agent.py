
from dotenv import load_dotenv
from langsmith import evaluate
from create_langchain_agent import agent

load_dotenv()


# Step 1: Run our Math Agent
def run_agent(inputs: dict) -> dict:

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": inputs["question"]
            }
        ]
    })

    return {
        "answer": response["messages"][-1].content
    }


# Step 2: Create a Code-Based Evaluator
def check_answer(outputs: dict, reference_outputs: dict) -> bool:

    actual_answer = str(outputs["answer"]).strip()
    expected_answer = str(reference_outputs["answer"]).strip()

    return actual_answer == expected_answer


# Step 3: Run the Experiment
results = evaluate(
    run_agent,
    data="math-agent-dataset",
    evaluators=[check_answer],
    experiment_prefix="math-agent-dataset-evaluation"
)

print("Evaluation completed!")
