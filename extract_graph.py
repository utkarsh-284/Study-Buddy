import time
import re
import json
import logging
from tqdm import tqdm
from openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader
from docling.document_converter import DocumentConverter
from langchain_text_splitters import RecursiveCharacterTextSplitter
import config

# -----------------------------
# Config
# -----------------------------
PDF_FILE = "./docs/WORLD_WAR_I.pdf"
OUTPUT_JSON_FILE = "history_graph_data.json" # File to save the output

# --- NVIDIA Client ---
NVIDIA_CLIENT = OpenAI(
  api_key=config.NVIDIA_API_KEY,
  base_url="https://integrate.api.nvidia.com/v1"
)
LLM_MODEL_NAME = "meta/llama-3.3-70b-instruct" #"meta/llama3-8b-instruct" # Using the more reliable 8B model

# Silence noisy loggers
logging.getLogger("openai").setLevel(logging.WARNING)

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
    * `"nodes"`: A list of objects. Example: `[ {{"id": "id_1", "label": "Concept", "name": "Name 1"}}, {{"id": "id_2", "label": "Person", "name": "Name 2"}} ]`
    * `"relationships"`: A list of objects. Example: `[ {{"source_id": "id_1", "target_id": "id_2", "type": "RELATED_TO"}}, {{"source_id": "chunk_id", "target_id": "id_1", "type": "MENTIONS"}} ]`

2.  **Node IDs:** Create simple, unique, lowercase_with_underscore IDs for all new entities.

3.  **Connect to Chunk:** This is most important. ALWAYS create a `DEFINES` or `MENTIONS` relationship **from the chunk (id: "{chunk_id}")** to every entity you find in it. This anchors the graph to the source text.

4.  **Connect Entities:** ALSO extract relationships *between* the entities themselves if they are clearly stated in the text. (e.g., {{"source_id": "photosynthesis", "target_id": "chloroplasts", "type": "TAKES_PLACE_IN"}}).

5.  **Focus:** Be precise and only extract information explicitly stated in the text. Prioritize quality and accuracy for a student's revision.

6.  **Critical:** Your response MUST be only the valid JSON object. Do not include any text, preamble, or explanations before or after the JSON.
"""


def extract_graph_from_chunk(client, text, chunk_id, retries=3, delay=5):
    """
    Uses an LLM to extract nodes and relationships from a text chunk.
    Includes robust retry and error handling.
    """
    prompt = EXTRACTION_PROMPT.format(
        text_chunk=text, 
        chunk_id=chunk_id
    )
    
    json_response = "" # Define in case of API error
    
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0
                # <-- The 'response_format' parameter has been removed
            )
            
            json_response = response.choices[0].message.content
            
            # --- THIS IS THE FIX ---
            # Find the start and end of the JSON object, ignoring any text
            # before or after it (like "Here is the JSON...").
            start_index = json_response.find('{')
            end_index = json_response.rfind('}')
            
            if start_index != -1 and end_index != -1:
                json_data = json_response[start_index : end_index + 1]
                graph = json.loads(json_data)
                return graph # Success!
            else:
                # The response did not contain a JSON object at all
                raise json.JSONDecodeError("No JSON object found in response.", json_response, 0)
            # --- END OF FIX ---

        # --- Specific Error Handling ---
        except json.JSONDecodeError as e:
            # This error is now the one we are protecting against
            # (e.g., if the model forgets a comma *even with the new prompt*)
            print(f"\n--- JSONDecodeError on attempt {attempt + 1} ---")
            print(f"Failed to decode LLM response: {e}")
            print(f"Raw response was: {json_response}")
            return {"nodes": [], "relationships": []}
        
        except Exception as e:
            # Your original, robust API error handling
            error_message = str(e)
            print(f"\n--- API Error on attempt {attempt + 1} ---")
            print(f"Error: {error_message}")
            
            is_auth_error = ("401" in error_message or "403" in error_message) and \
                            ("permission" in error_message.lower() or "authentication" in error_message.lower())
            
            is_timeout = "504" in error_message or "Gateway Timeout" in error_message
            is_rate_limit = "429" in error_message or "rate limit" in error_message.lower()

            if is_timeout:
                if attempt < retries - 1:
                    print(f"Gateway Timeout. Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    print("Max retries reached. Failing this chunk.")
            
            elif is_auth_error:
                print("CRITICAL: Authentication or Permission Error. Your API key is not authorized for this model.")
                return {"nodes": [], "relationships": []} # Fatal error

            elif is_rate_limit:
                print(f"Rate limit hit. Retrying in {delay * 2} seconds...")
                time.sleep(delay * 2) # Longer delay
            
            else:
                print(f"Non-retryable error: {e}")
                return {"nodes": [], "relationships": []}
                
    return {"nodes": [], "relationships": []}


# -----------------------------
# Main Function
# -----------------------------
def main():
    # 1. Load and Chunk the PDF
    print(f"Loading document: {PDF_FILE}...")
    loader = PyPDFLoader(PDF_FILE)
    pages = loader.load()
    
    # Remove references section
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
        chunk_size=1024, # Larger chunks for better context
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(cleaned_pages)
    print(f"Document chunked into {len(chunks)} pieces.")

    # --- Lists to hold all extracted data ---
    all_nodes = []
    all_relationships = []

    # 2. Process each chunk
    for i, chunk in enumerate(tqdm(chunks, desc="Extracting Graph from Chunks")):
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
        all_nodes.append(chunk_node)
        
        # 4. Extract graph from chunk text using LLM
        graph = extract_graph_from_chunk(NVIDIA_CLIENT, chunk_text, chunk_id)
        
        # 5. Add extracted data to our lists
        all_nodes.extend(graph.get("nodes", []))
        all_relationships.extend(graph.get("relationships", []))

        # 6. Add 1 second delay to avoid rate limiting
        time.sleep(1)
    # 6. Save all data to a single JSON file
    final_graph_data = {
        "nodes": all_nodes,
        "relationships": all_relationships
    }
    
    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(final_graph_data, f, indent=2, ensure_ascii=False)

    print(f"\nSuccessfully extracted all data and saved to {OUTPUT_JSON_FILE}")

# -----------------------------
if __name__ == "__main__":
    main()