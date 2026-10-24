
from dotenv import load_dotenv
from langsmith import Client

load_dotenv()

client = Client()

DATASET_NAME = "simple-math-tests"

# Define dataset examples
examples = [
    {
        "inputs": {"question": "What is 5 + 3?"},
        "outputs": {"answer": "8"}
    },
    {
        "inputs": {"question": "What is 10 + 20?"},
        "outputs": {"answer": "30"}
    },
    {
        "inputs": {"question": "What is 7 + 2?"},
        "outputs": {"answer": "9"}
    },
    {
        "inputs": {"question": "What is 15 + 5?"},
        "outputs": {"answer": "20"}
    }
]

# Step 1: Check if dataset already exists
if client.has_dataset(dataset_name=DATASET_NAME):

    print("Dataset already exists. Replacing it...")

    # Delete existing dataset
    client.delete_dataset(dataset_name=DATASET_NAME)

# Step 2: Create dataset
dataset = client.create_dataset(
    dataset_name=DATASET_NAME,
    description="Math questions for testing our agent"
)

# Step 3: Add examples
client.create_examples(
    dataset_id=dataset.id,
    examples=examples
)

print("Dataset created/replaced successfully!")
print("Dataset Name:", dataset.name)
print("Dataset ID:", dataset.id)
