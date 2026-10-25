
import os
from dotenv import load_dotenv
from langsmith import Client

load_dotenv(override=True)

client = Client()

print("API Endpoint:", client.api_url)
print("Workspace ID:", client.workspace_id or "Not explicitly set")

# Check the dataset that Python created
dataset = client.read_dataset(
    dataset_id="f44229a4-da21-46ca-8887-609754c2dd38"
)

print("\nDataset:", dataset.name)
print("Dataset URL:", dataset.url)

# Check whether the existing Math dataset
# is accessible through the same connection
print("\nMath dataset accessible:",
      client.has_dataset(dataset_name="math-agent-dataset"))
