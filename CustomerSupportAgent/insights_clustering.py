
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

client = Client()

# 1. Read traces from your existing project
runs = list(client.list_runs(
    project_name="langchain-project",
    is_root=True,
    limit=100
))

# 2. Keep only our DemoShop customer requests
demoshop_runs = [
    run for run in runs
    if run.name == "DemoShop_Customer_Request"
    and run.outputs
    and "question" in (run.inputs or {})
]

print("DemoShop traces found:", len(demoshop_runs))

if not demoshop_runs:
    raise ValueError(
        "No DemoShop traces found. "
        "Run the DemoShop agent with RUN_DEMO_BATCH=True first."
    )

# 3. Convert traces into conversation histories
chat_histories = []

for run in demoshop_runs:
    question = run.inputs["question"]
    answer = run.outputs.get("output", "")

    chat_histories.append([
        {"role": "user", "content": str(question)},
        {"role": "assistant", "content": str(answer)}
    ])

# 4. Generate clustering report
report = client.generate_insights(
    chat_histories=chat_histories,
    name="DemoShop Customer Clustering",
    instructions=(
        "Group customer requests into meaningful categories "
        "such as order tracking, inventory, returns, shipping, "
        "and unsupported requests. Identify common problems."
    ),
    model="openai",
    openai_api_key=os.environ["OPENAI_API_KEY"]
)

print("Clustering report submitted!")
print("Report URL:", report.link)
