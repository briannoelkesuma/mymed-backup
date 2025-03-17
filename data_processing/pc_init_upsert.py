from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import sys
import os
from pathlib import Path

# Add parent directory to Python path to fix import
sys.path.append(str(Path(__file__).parent.parent))

from data_processing.pc_embeddings import upload_embeddings_to_pinecone
import json

load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Set up Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to Pinecone index
index_name = "mymed-allparams"

# Check if the index exists, and create if it doesn’t
if index_name not in pc.list_indexes():
    print(f"Creating Pinecone index '{index_name}' as it does not exist.")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"Pinecone index '{index_name}' already exists.")

# Connect to the Pinecone index
pinecone_index = pc.Index(index_name)
print(f"Connected to Pinecone index '{index_name}'.")

# Upload embeddings to Pinecone
with open('../scraping/scraped_data_allparams.json', 'r') as file:
    documents = json.load(file)

upload_embeddings_to_pinecone(documents, pinecone_index)