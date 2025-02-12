import uuid
from nltk.tokenize import sent_tokenize
# from llama_index.embeddings.openai import OpenAIEmbedding
from langchain_openai import OpenAIEmbeddings
from backend.settings import OPENAI_API_KEY
import json
from dotenv import load_dotenv
import os
from data_processing.chunk_processing import split_markdown_sections, chunk_sections, process_markdown
import uuid

load_dotenv()

# Load API keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY, model='text-embedding-3-small')

def upload_embeddings_to_pinecone(documents, pinecone_index):
    for page in documents:
        print(f"Processing page: {page['metadata']['title']}")
        processed_chunks = process_markdown(page)
        for i, chunk in enumerate(processed_chunks):
            text_content = chunk["content"]  # Extract text content for embedding
            embedding_vector = embedding_model.embed_query(text_content)  # Generate embedding

            unique_id = f"{chunk['id']}-{uuid.uuid4().hex[:8]}"  # Generate unique ID

            # Define metadata
            metadata = {
                "title_chunk_id": chunk['id'],
                "title": chunk["title"],
                "source_url": chunk["source_url"],
                "content": chunk["content"],
                "page_title": chunk["metadata"]["page_title"],
                "page_description": chunk["metadata"]["page_description"]
            }

            # Upsert into Pinecone (ID, vector, metadata)
            pinecone_index.upsert(
                [(unique_id, embedding_vector, metadata)]
            )
            print(f"Uploaded chunk {i} from {chunk['source_url']} to Pinecone.")

    print("Embeddings successfully uploaded to Pinecone!")