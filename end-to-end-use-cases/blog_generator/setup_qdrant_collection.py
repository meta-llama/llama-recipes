


"""
Script to set up a Qdrant collection with provided markdown files.
To use this script, replace the file paths in the NEW_COLLECTIONS list with your own markdown files.
Then, run the script using Python: `python setup_qdrant_collection.py`
"""



from pathlib import Path
import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import uuid
import re

# Configuration - in case you want to create an online collection
QDRANT_URL = os.environ.get("QDRANT_URL", "")
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY", "")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'

# New files to process
# IMPORTANT: Added the configuration for readme_blogs_latest here
NEW_COLLECTIONS = [
    {
        "file_path": "path/to/your/markdown/file1.txt",
        "collection_name": "example_collection_1"
    },
    {
        "file_path": "path/to/your/markdown/file2.txt",
        "collection_name": "example_collection_2"
    }

]

def markdown_splitter(text, max_chunk=800):
    sections = re.split(r'(?=^#+ .*)', text, flags=re.MULTILINE)
    chunks = []
    current_chunk = []
    
    for section in sections:
        if len(''.join(current_chunk)) + len(section) > max_chunk:
            chunks.append(''.join(current_chunk))
            current_chunk = [section]
        else:
            current_chunk.append(section)
    
    if current_chunk:
        chunks.append(''.join(current_chunk))
    
    return [{"text": chunk, "header": f"section_{i}"} for i, chunk in enumerate(chunks)]

def get_qdrant_client():
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def get_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)

def process_file(config):
    client = get_qdrant_client()
    embedding_model = get_embedding_model()
    
    # Create collection if not exists
    if not client.collection_exists(config["collection_name"]):
        client.create_collection(
            collection_name=config["collection_name"],
            vectors_config=models.VectorParams(
                size=384,
                distance=models.Distance.COSINE
            )
        )
    
    # Process and store documents
    try:
        text = Path(config["file_path"]).read_text(encoding='utf-8')
        chunks = markdown_splitter(text)
        
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            points = []
            for chunk in batch:
                embedding = embedding_model.encode(chunk["text"]).tolist()
                points.append(
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=embedding,
                        payload=chunk
                    )
                )
            client.upsert(collection_name=config["collection_name"], points=points)
        
        print(f"Processed {len(chunks)} chunks for {config['collection_name']}")
    except FileNotFoundError:
        print(f"Error: The file at {config['file_path']} was not found. Skipping collection setup.")

def setup_all_collections():
    for config in NEW_COLLECTIONS:
        process_file(config)
    print("All collections created and populated successfully!")

if __name__ == "__main__":
    setup_all_collections()