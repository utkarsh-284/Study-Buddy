import time
import re
import json
import logging
from tqdm import tqdm
from openai import OpenAI
from neo4j import GraphDatabase
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# -----------------------------
# Config
# -----------------------------
PDF_FILE = "./docs/01_Attention_Is_All_You_Need.pdf" # The PDF to process
BATCH_SIZE = 32 # For Pinecone (if you run both)
EMBED_MODEL_NAME = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
LLM_MODEL_NAME = "meta/llama-3.3-70b-instruct" # Used for extraction


# --- Initilaize Client ---
# Used for graph extraction
client = OpenAI(
  api_key=config.NVIDIA_API_KEY,
  base_url="https://integrate.api.nvidia.com/v1"
)


# --- Neo4j Client ---
DRIVER = GraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# Silence noisy loggers
logging.getLogger("pinecone").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("neo4j").setLevel(logging.WARNING)

# -----------------------------
# Graph Extraction
# -----------------------------

EXTRACTION_PROMPT = """
You are an AI-powered learning assistant. Your goal is to convert unstructured study notes 
into a structured knowledge graph for revision. The graph should capture the main ideas,
defined terms, and how they relate to each other.

Read the following text chunk and extract all key entities and their relationships based 
on the schema below.

**Schema:**

* **Nodes (Entities):**
    * `Concept`: A high-level idea, topic, or theory. (e.g., "Photosynthesis", "Feudalism", "Recursion", "The Renaissance").
    * `Term`: A specific, defined word or phrase; a piece of vocabulary. (e.g., "Chlorophyll", "Vassal", "Base Case").
    * `Person`: An individual. (e.g., "Marie Curie", "King Louis XIV", "Guido van Rossum").
    * `Place`: A location, real or abstract. (e.g., "Mitochondria", "Versailles", "The CPU").
    * `Event`: A specific, significant happening. (e.g., "French Revolution", "Compiler Error").

* **Relationships (Edges):**
    * `DEFINES`: The chunk provides a definition for a Term or Concept. 
        (e.g., "{chunk_id}" -[DEFINES]-> "vassal")
    * `MENTIONS`: The chunk mentions an entity without defining it.
        (e.g., "{chunk_id}" -[MENTIONS]-> "king_louis_xiv")
    * `IS_A`: A hierarchical link. 
        (e.g., "Vassal" -[IS_A]-> "Person", "Recursion" -[IS_A]-> "Concept")
    * `PART_OF`: A component link.
        (e.g., "Chlorophyll" -[PART_OF]-> "Chloroplast")
    * `EXAMPLE_OF`: Identifies a concrete example of a concept.
        (e.g., "A 'for loop'" -[EXAMPLE_OF]-> "Iteration")
    * `CAUSES` / `LEADS_TO`: A causal or sequential link.
        (e.g., "Event A" -[CAUSES]-> "Event B")
    * `TAKES_PLACE_IN`: A link between an Event/Concept and a Place.
        (e.g., "French Revolution" -[TAKES_PLACE_IN]-> "France")
    * `RELATED_TO`: A generic connection when no other type fits.

**Input Text:**
"{text_chunk}"

**Instructions:**

1.  **JSON Output:** Return a single JSON object with two keys: "nodes" and "relationships".
    * `"nodes"`: A list of objects. Format: `[ {{"id": "...", "label": "Concept", "name": "..."}} ]`
    * `"relationships"`: A list of objects. Format: `[ {{"source_id": "...", "target_id": "...", "type": "..."}} ]`

2.  **Node IDs:** Create simple, unique, lowercase_with_underscore IDs for all new entities.

3.  **Connect to Chunk:** This is most important. ALWAYS create a `DEFINES` or `MENTIONS` relationship **from the chunk (id: "{chunk_id}")** to every entity you find in it. This anchors the graph to the source text.

4.  **Connect Entities:** ALSO extract relationships *between* the entities themselves if they are clearly stated in the text. (e.g., {{"source_id": "photosynthesis", "target_id": "chloroplasts", "type": "TAKES_PLACE_IN"}}).

5.  **Focus:** Be precise and only extract information explicitly stated in the text. Prioritize quality and accuracy for a student's revision.
"""

def extract_graph_from_chunk(client, text, chunk_id):
    """
    Uses an LLM to extract nodes and relationships from a text chunk.
    """
    # Note: We removed paper_id, it's no longer needed
    prompt = EXTRACTION_PROMPT.format(
        text_chunk=text, 
        chunk_id=chunk_id
    )
    
    try:
        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        
        json_response = response.choices[0].message.content
        graph = json.loads(json_response)
        return graph
        
    except Exception as e:
        print(f"Error in LLM extraction: {e}")
        return {"nodes": [], "relationships": []}

# -----------------------------
# Neo4j Helper Functions (Modified)
# -----------------------------
def create_constraints(tx):
    """Create uniqueness constraints for all expected node labels."""
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (m:Method) REQUIRE m.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Chunk) REQUIRE k.id IS UNIQUE")

def upsert_node(tx, node):
    """
    Upsert a node given its id, label, and other properties.
    The 'node' param is a dict, e.g.:
    {"id": "transformer", "label": "Concept", "name": "Transformer"}
    """
    labels = [node.get("label", "Entity"), "Entity"] # Always add :Entity
    label_cypher = ":" + ":".join(set(labels)) # Ensure unique labels
    
    # All properties *except* 'label' (which defines the node type)
    props = {k:v for k,v in node.items() if k != "label"}
    
    cypher = (
        f"MERGE (n{label_cypher} {{id: $id}}) "
        "SET n += $props"
    )
    tx.run(cypher, id=node["id"], props=props)

def create_relationship(tx, rel):
    """
    Create a relationship given a source id, target id, and type.
    The 'rel' param is a dict, e.g.:
    {"source_id": "chunk_1", "target_id": "transformer", "type": "MENTIONS_CONCEPT"}
    """
    rel_type = rel.get("type", "RELATED_TO").upper().replace(" ", "_")
    source_id = rel.get("source_id")
    target_id = rel.get("target_id")
    
    if not source_id or not target_id:
        return

    cypher = (
        "MATCH (a:Entity {id: $source_id}), (b:Entity {id: $target_id}) "
        f"MERGE (a)-[r:{rel_type}]->(b) "
        "RETURN r"
    )
    tx.run(cypher, source_id=source_id, target_id=target_id)

# -----------------------------
# Main Function (Modified)
# -----------------------------
def main():
    # 1. Load and Chunk the PDF
    print(f"Loading document: {PDF_FILE}...")
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()
    
    # (Optional) Remove references section
    cleaned_pages = []
    found_references = False
    ref_pattern = re.compile(r'^(References|REFERENCES)$', re.MULTILINE)
    for page in pages:
        if found_references: continue
        search_result = ref_pattern.search(page.page_content)
        if search_result:
            found_references = True
            cleaned_content = page.page_content[:search_result.start()]
            if cleaned_content.strip():
                page.page_content = cleaned_content
                cleaned_pages.append(page)
        else:
            cleaned_pages.append(page)
    
    print(f"Document loaded. Total pages (pre-references): {len(cleaned_pages)}")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024, # Keep larger chunks for better context
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(cleaned_pages)
    print(f"Document chunked into {len(chunks)} pieces.")

    with DRIVER.session() as session:
        # Create constraints first
        session.execute_write(create_constraints)
        
        # 2. Process each chunk
        for i, chunk in enumerate(tqdm(chunks, desc="Extracting & Loading Graph")):
            # Create a unique chunk ID based on the file and index
            chunk_id = f"{PDF_FILE}_chunk_{i}"
            chunk_text = chunk.page_content
            page_num = chunk.metadata.get("page", 0) + 1

            # 3. Create the Chunk node
            chunk_node = {
                "id": chunk_id,
                "label": "Chunk",
                "text": chunk_text,
                "page_number": page_num,
                "source_file": PDF_FILE
            }
            session.execute_write(upsert_node, chunk_node)
            
            # 4. Extract graph from chunk text using LLM
            # Pass only the chunk_id
            graph = extract_graph_from_chunk(client, chunk_text, chunk_id)
            
            # 5. Load extracted graph into Neo4j
            for node in graph.get("nodes", []):
                session.execute_write(upsert_node, node)
            
            for rel in graph.get("relationships", []):
                session.execute_write(create_relationship, rel)

    print("All items uploaded successfully to Neo4j.")
    DRIVER.close()

# -----------------------------
if __name__ == "__main__":
    main()