import json
import logging
from tqdm import tqdm
from neo4j import GraphDatabase
import config

# -----------------------------
# Config
# -----------------------------
INPUT_JSON_FILE = "graph_data.json"

# --- Neo4j Client ---
DRIVER = GraphDatabase.driver(
    config.NEO4J_URI, 
    auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# Silence noisy loggers
logging.getLogger("neo4j").setLevel(logging.WARNING)

# -----------------------------
# Neo4j Helper Functions
# -----------------------------
def create_constraints(tx):
    """
    Create uniqueness constraints for all expected node labels.
    We add :Entity as a base label for all nodes for a global
    uniqueness constraint on 'id'.
    """
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Entity) REQUIRE n.id IS UNIQUE")
    
    # --- ADDED NEW LABELS ---
    # Specific constraints for your new schema
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (c:Concept) REQUIRE c.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (k:Chunk) REQUIRE k.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (t:Term) REQUIRE t.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (e:Event) REQUIRE e.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (l:Place) REQUIRE l.id IS UNIQUE")
    tx.run("CREATE CONSTRAINT IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE")
    # --------------------------


def upsert_node(tx, node):
    """
    Upsert a node given its id, label, and other properties.
    This function will MERGE on the node's 'id' ONLY,
    then SET all other properties, including the label.
    """
    
    # --- THIS IS THE FIX ---
    # Add a unique Project label for every node
    PROJECT_LABEL = "StudyBuddy"

    # Get all properties from the node dict
    props = node.copy()
    node_id = props.get("id")
    
    # Get the primary label (e.g., "Concept", "Chunk", "Person")
    # Default to "Entity" if no label is provided
    labels = [node.get("label", "Entity"), "Entity", PROJECT_LABEL]
    # Create the label string, e.g., ":Concept:Entity"
    label_cypher = ":" + ":".join(set(l for l in labels if l))

    cypher = (
        f"MERGE (n:Entity {{id: $id}}) "  # 1. Find or create the node by ID ONLY
        f"SET n = $props "                 # 2. Overwrite all properties
        f"SET n{label_cypher}"             # 3. Set/update all labels
    )
    
    tx.run(cypher, id=node_id, props=props)
    # --- END OF FIX ---


def create_relationship(tx, rel):
    """
    Create a relationship given a source id, target id, and type.
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
# Main Function
# -----------------------------
def main():
    print(f"Loading graph data from {INPUT_JSON_FILE}...")
    try:
        with open(INPUT_JSON_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: {INPUT_JSON_FILE} not found.")
        print("Please run `extract_graph_from_pdf.py` first.")
        return
    
    nodes = data.get("nodes", [])
    relationships = data.get("relationships", [])
    
    print(f"Found {len(nodes)} nodes and {len(relationships)} relationships.")

    with DRIVER.session() as session:
        # Create constraints first
        print("Ensuring database constraints...")
        session.execute_write(create_constraints)
        
        # 2. Upload Nodes
        print("Uploading nodes...")
        for node in tqdm(nodes, desc="Uploading Nodes"):
            if not node.get("id"):
                print(f"Skipping node with no ID: {node}")
                continue
            session.execute_write(upsert_node, node)
            
        # 3. Upload Relationships
        print("Uploading relationships...")
        for rel in tqdm(relationships, desc="Uploading Relationships"):
            session.execute_write(create_relationship, rel)

    print("\nAll items uploaded successfully to Neo4j.")
    DRIVER.close()

# -----------------------------
if __name__ == "__main__":
    main()