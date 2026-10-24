
from dotenv import load_dotenv
from langsmith import evaluate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from create_langchain_agent import agent

load_dotenv()


# -----------------------------------
# Step 1: Run the existing Math Agent
# -----------------------------------

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


# -----------------------------------
# Step 2: Define the Judge output
# -----------------------------------

class JudgeResult(BaseModel):
    correct: bool = Field(
        description="True if the answer is mathematically correct"
    )
    reason: str = Field(
        description="Short explanation of the score"
    )


# -----------------------------------
# Step 3: Create the Judge LLM
# -----------------------------------

judge = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).with_structured_output(JudgeResult)


# -----------------------------------
# Step 4: Create LLM-as-Judge evaluator
# -----------------------------------

def llm_judge(inputs: dict,
              outputs: dict,
              reference_outputs: dict) -> dict:

    question = inputs["question"]
    actual = outputs["answer"]
    expected = reference_outputs["answer"]

    result = judge.invoke([
        (
            "system",
            "You are a math evaluation judge. "
            "Check whether the agent's answer is "
            "mathematically equivalent to the expected answer. "
            "Ignore wording differences. "
            "Provide a short explanation."
        ),
        (
            "human",
            f"Question: {question}\n"
            f"Expected Answer: {expected}\n"
            f"Agent Answer: {actual}"
        )
    ])

    return {
        "key": "llm_correctness",
        "score": int(result.correct),
        "comment": result.reason
    }


# -----------------------------------
# Step 5: Run the experiment
# -----------------------------------

results = evaluate(
    run_agent,
    data="math-agent-dataset",
    evaluators=[llm_judge],
    experiment_prefix="math-agent-llm-judge"
)

print("LLM-as-Judge evaluation completed!")
