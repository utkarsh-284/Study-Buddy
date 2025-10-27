import time
import logging
import re   # Importing Regular expression
from tqdm import tqdm
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import config

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# ------------------------------------

# --- 1. LOGGING FIX: Silence noisy loggers ---
# This will stop logs from breaking the progress bar
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
# -------------------------------------------

# -----------------------------
# Config
# -----------------------------
PDF_FILE = "./docs/01_Attention_Is_All_You_Need.pdf"
BATCH_SIZE = 16

# Model and Vector Config
EMBED_MODEL_NAME = "nomic-embed-text:latest"
VECTOR_DIM = 768

# Pinecone Config
INDEX_NAME = config.PINECONE_INDEX_NAME

# -----------------------------
# Initialize clients
# -----------------------------
print("Initializing clients...")
try:
    embeddings_client = OllamaEmbeddings(model=EMBED_MODEL_NAME)
    print(f"Ollama embeddings initialized with model: {EMBED_MODEL_NAME}")
except Exception as e:
    print(f"Failed to initialize Ollama. Is Ollama running? \nError: {e}")
    exit()

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME} with dimension {VECTOR_DIM}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    while not pc.describe_index(INDEX_NAME).status['ready']:
        print("Waiting for index to be ready...")
        time.sleep(5)
else:
    index_desc = pc.describe_index(INDEX_NAME)
    if index_desc.dimension != VECTOR_DIM:
        print(f"ERROR: Index {INDEX_NAME} exists but has dimension {index_desc.dimension}. Expected {VECTOR_DIM}.")
        print("Please delete the index or update VECTOR_DIM and INDEX_NAME.")
        exit()
    print(f"Index {INDEX_NAME} already exists.")

index = pc.Index(INDEX_NAME)
print(f"Connected to index {INDEX_NAME}.")

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts, client=embeddings_client):
    # Use LangChain's batch embedding method
    return client.embed_documents(texts)

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    # ---1. Loading Document Use PyPDFLoader ---
    print(f"Loading document: {PDF_FILE} using PyPDFLoader...")
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load() # Loads 1 Document per page
    print(f"Document loaded. Total pages: {len(pages)}")

    # === 2. FIND AND REMOVE REFERECES ===
    cleaned_pages = []
    found_refereces = False
    # This regex looks for "References" or "REFERENCES" on its own line
    split_pattern = re.compile(r'^(References|REFERENCES)$', re.MULTILINE | re.IGNORECASE)

    for page in pages:
        if found_refereces:
            # If we've already found the ref section, skip all subsequent pages
            continue

        search_result = split_pattern.search(page.page_content)

        if search_result:
            # Found the "References" heading on this page
            found_refereces = True

            # Get the text **before** the match
            start_index_of_match = search_result.start()
            cleaned_content = page.page_content[:start_index_of_match]

            if cleaned_content.strip():
                # If there's meaningful content before "References", keep it
                page.page_content = cleaned_content
                cleaned_pages.append(page)
        
        else:
            # This page is before the references, keep it as is
            cleaned_pages.append(page)

    if not found_refereces:
        print("Warning: 'References' section not found. Indexing full document.")
        cleaned_pages = pages # Use all pages if no reference was found
    else:
        print(f"Removed 'References' section. Usable pages: {len(cleaned_pages)}")


    # 2. Chunk the document with RecursiveCharacterTextSplitter
    print("Chunking document with RecursiveCharacterTextSplitter...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50,
        length_function=len,
    )
    chunks = text_splitter.split_documents(cleaned_pages) # Splits cleaned_pages into chunks
    print(f"Document chunked into {len(chunks)} pieces.")

    # 3. Prepare items for Pinecone
    items = []
    for i, chunk in enumerate(chunks):
        text = chunk.page_content
        
        # --- 2. LOADER FIX: Get correct page number ---
        # PyPDFLoader stores page number in 'page' (0-indexed)
        # We add 1 to make it 1-indexed (optional, but more human-readable)
        page_num = chunk.metadata.get("page", 0) + 1
        # ----------------------------------------------
        
        meta = {
            "filename": chunk.metadata.get("source"),
            "page_number": page_num,
            "text_snippet": text[:200]
        }
        
        chunk_id = f"{PDF_FILE}-chunk-{i}"
        
        items.append((chunk_id, text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    # 4. Batch, embed, and upsert
    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")
    print(index.describe_index_stats())

# -----------------------------
if __name__ == "__main__":
    main()