
from dotenv import load_dotenv
from langsmith import evaluate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# Reuse our existing agent runner
from evaluate_demoshop_code import run_agent

load_dotenv()

DATASET_NAME = "demoshop-support-dataset"


# ==========================================
# 1. DEFINE JUDGE RESPONSE STRUCTURE
# ==========================================

class JudgeResult(BaseModel):

    correct: bool = Field(
        description="Whether the agent answered correctly"
    )

    reason: str = Field(
        description="Short explanation of the score"
    )


# ==========================================
# 2. CREATE LLM JUDGE
# ==========================================

judge = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
).with_structured_output(JudgeResult)


# ==========================================
# 3. CREATE LLM-AS-JUDGE EVALUATOR
# ==========================================

def llm_judge(
    inputs: dict,
    outputs: dict,
    reference_outputs: dict
) -> dict:

    question = inputs["question"]
    actual_answer = outputs["answer"]
    expected_answer = reference_outputs["answer"]

    verdict = judge.invoke([
        (
            "system",
            "You are evaluating an online-store customer-support "
            "agent. Decide whether its answer correctly addresses "
            "the customer's question using the reference answer "
            "as the source of truth. Different wording is fine. "
            "Mark incorrect if it misses important facts, "
            "contradicts the reference, or invents information. "
            "Give a short explanation."
        ),
        (
            "human",
            f"Customer question: {question}\n"
            f"Expected answer: {expected_answer}\n"
            f"Agent answer: {actual_answer}"
        )
    ])

    return {
        "key": "llm_answer_correctness",
        "score": int(verdict.correct),
        "comment": verdict.reason
    }


# ==========================================
# 4. RUN EVALUATION
# ==========================================

if __name__ == "__main__":

    evaluate(
        run_agent,
        data=DATASET_NAME,
        evaluators=[llm_judge],
        experiment_prefix="demoshop-llm-judge",
        max_concurrency=2
    )

    print("LLM-as-Judge evaluation complete!")
