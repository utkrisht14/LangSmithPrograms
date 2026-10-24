from langsmith import Client
from dotenv import load_dotenv
load_dotenv()

client = Client()
dataset = client.create_dataset(
dataset_name="simple-math-tests"
)
client.create_examples(
dataset_id=dataset.id,
examples=[{
"inputs": {"question": "5 + 5?"},
"outputs": {"answer": "10"}
}]
)