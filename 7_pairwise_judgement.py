
import re
from typing import Literal

from dotenv import load_dotenv
from langsmith import evaluate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_NAME = "math-agent-dataset"

# True = Compare existing experiments
# False = Run both agents again
REUSE_EXISTING_EXPERIMENTS = True

CODE_EXPERIMENT = "code-agent-91f84793"
LLM_EXPERIMENT = "llm-agent-2a2ebbff"


# ==========================================
# 1. CODE-BASED AGENT
# ==========================================

def code_agent(inputs: dict) -> dict:

    question = inputs["question"]

    match = re.search(
        r"(-?\d+)\s*\+\s*(-?\d+)",
        question
    )

    if not match:
        return {"answer": "Unsupported question"}

    a = int(match.group(1))
    b = int(match.group(2))

    return {"answer": str(a + b)}


# ==========================================
# 2. LLM-BASED AGENT
# ==========================================

def llm_agent(inputs: dict) -> dict:

    from create_langchain_agent import agent

    response = agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": inputs["question"]
            }
        ]
    })

    return {
        "answer": str(response["messages"][-1].content)
    }


# ==========================================
# 3. HELPER: EXTRACT ANSWER SAFELY
# ==========================================

def extract_answer(data):

    if data is None:
        raise ValueError("Agent output is None")

    if isinstance(data, (str, int, float)):
        return str(data).strip()

    if isinstance(data, dict):

        # Support different LangSmith output shapes
        for key in (
            "answer",
            "output",
            "outputs",
            "response",
            "result",
            "content",
            "text"
        ):
            if key in data:
                return extract_answer(data[key])

        if "messages" in data:
            return extract_answer(data["messages"])

        raise ValueError(
            f"Cannot find answer in output: {data}"
        )

    if isinstance(data, list):

        if not data:
            raise ValueError("Empty output list")

        return extract_answer(data[-1])

    raise TypeError(
        f"Unsupported output type: {type(data)}"
    )


# ==========================================
# 4. JUDGE OUTPUT STRUCTURE
# ==========================================

class JudgeResult(BaseModel):

    winner: Literal["A", "B", "tie"] = Field(
        description="The better response or tie"
    )

    reason: str = Field(
        description="Short reason for the decision"
    )


# ==========================================
# 5. CREATE LLM JUDGE
# ==========================================

judge = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).with_structured_output(JudgeResult)


# ==========================================
# 6. PAIRWISE EVALUATOR
# ==========================================

def pairwise_judge(runs, example) -> dict:

    if len(runs) != 2:
        raise ValueError("Expected exactly two runs")

    run_a, run_b = runs

    # Extract outputs from both experiments
    answer_a = extract_answer(run_a.outputs)
    answer_b = extract_answer(run_b.outputs)

    # Dataset input and expected answer
    question = example.inputs["question"]
    expected = extract_answer(example.outputs)

    print("\n------------------------------")
    print("Question:", question)
    print("Expected:", expected)
    print("Response A:", answer_a)
    print("Response B:", answer_b)

    # Ask the LLM Judge
    result = judge.invoke([
        (
            "system",
            """
            You are an impartial math evaluator.

            Compare two responses to the same math question.

            Evaluation criteria:

            1. Mathematical correctness is most important.
            2. Prefer a correct answer over an incorrect one.
            3. If both are correct, prefer clarity and conciseness.
            4. If both are equally good, return tie.
            5. Do not favor A or B because of position.

            Return the winner and a short reason.
            """
        ),
        (
            "human",
            f"Question: {question}\n"
            f"Expected Answer: {expected}\n"
            f"Response A: {answer_a}\n"
            f"Response B: {answer_b}"
        )
    ])

    print("Judge Winner:", result.winner)
    print("Judge Reason:", result.reason)

    # Convert judge decision into scores
    if result.winner == "A":
        score_a, score_b = 1, 0

    elif result.winner == "B":
        score_a, score_b = 0, 1

    else:
        score_a, score_b = 0, 0

    # Return LangSmith comparative feedback
    return {
        "key": "pairwise_math_judge",

        "scores": {
            str(run_a.id): score_a,
            str(run_b.id): score_b
        },

        "comment": result.reason
    }


# ==========================================
# 7. MAIN FUNCTION
# ==========================================

def main():

    if REUSE_EXISTING_EXPERIMENTS:

        print("Using existing experiments...")

        experiment_a = CODE_EXPERIMENT
        experiment_b = LLM_EXPERIMENT

    else:

        print("Running Code-Based Agent...")

        code_results = evaluate(
            code_agent,
            data=DATASET_NAME,
            experiment_prefix="code-agent"
        )

        print("Running LLM-Based Agent...")

        llm_results = evaluate(
            llm_agent,
            data=DATASET_NAME,
            experiment_prefix="llm-agent"
        )

        experiment_a = code_results.experiment_name
        experiment_b = llm_results.experiment_name

    # ======================================
    # 8. RUN PAIRWISE EVALUATION
    # ======================================

    print("\nStarting Pairwise Evaluation...")

    results = evaluate(
        (
            experiment_a,
            experiment_b
        ),
        evaluators=[pairwise_judge],
        randomize_order=True,
        experiment_prefix="math-pairwise-fixed",
        max_concurrency=1
    )

    print("\nPairwise Evaluation Completed!")
    print("Check LangSmith Pairwise Experiments.")


# ==========================================
# 9. EXECUTE
# ==========================================

if __name__ == "__main__":
    main()
