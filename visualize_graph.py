# visualize_graph.py
from neo4j import GraphDatabase
from pyvis.network import Network
import config
import re

NEO_BATCH = 500  # number of relationships to fetch / visualize

driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD))

def fetch_subgraph(tx, limit=500):
    """
    Fetch all nodes and relationships up to a limit
    that ONLY belong to our project.
    """
    # We now match on (a:StudyBuddy) instead of (a:Entity)
    q = (
        "MATCH (a:StudyBuddy)-[r]->(b:StudyBuddy) "
        "RETURN a.id AS a_id, labels(a) AS a_labels, a.name AS a_name, "
        "b.id AS b_id, labels(b) AS b_labels, b.name AS b_name, type(r) AS rel "
        "LIMIT $limit"
    )
    return list(tx.run(q, limit=limit))

def build_pyvis(rows, output_html="neo4j_viz.html"):
    """
    Builds the Pyvis network graph from the Neo4j query results.
    """
    net = Network(height="900px", width="100%", notebook=False, directed=True, 
                  bgcolor="#222222", font_color="white")
    
    # Add some physics options for better layout
    net.set_options("""
    var options = {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -20000,
          "centralGravity": 0.3,
          "springLength": 200,
          "springConstant": 0.04
        },
        "minVelocity": 0.75
      }
    }
    """)

    for rec in rows:
        a_id = rec["a_id"]; a_labels = rec["a_labels"]
        b_id = rec["b_id"]; b_labels = rec["b_labels"]
        rel = rec["rel"]

        # --- MODIFICATION ---
        # Shorten Chunk labels for readability
        if "Chunk" in a_labels:
            # Use regex to find 'chunk_X'
            match = re.search(r'chunk_(\d+)', a_id)
            a_name = f"Chunk {match.group(1)}" if match else a_id
            a_label = f"{a_name}\n({','.join(a_labels)})"
            a_title = a_id # Show full ID on hover
            a_color = "#C7A9DC" # Give Chunks a distinct color
        else:
            a_name = rec["a_name"] or a_id
            a_label = f"{a_name}\n({','.join(a_labels)})"
            a_title = a_name
            a_color = "#68B0D6" # Color for Concepts, Persons, etc.

        if "Chunk" in b_labels:
            match = re.search(r'chunk_(\d+)', b_id)
            b_name = f"Chunk {match.group(1)}" if match else b_id
            b_label = f"{b_name}\n({','.join(b_labels)})"
            b_title = b_id
            b_color = "#C7A9DC"
        else:
            b_name = rec["b_name"] or b_id
            b_label = f"{b_name}\n({','.join(b_labels)})"
            b_title = b_name
            b_color = "#68B0D6"
        # --- END MODIFICATION ---

        net.add_node(a_id, label=a_label, title=a_title, color=a_color)
        net.add_node(b_id, label=b_label, title=b_title, color=b_color)
        net.add_edge(a_id, b_id, title=rel)

    net.show(output_html, notebook=False)
    print(f"Saved visualization to {output_html}")

def main():
    print("Connecting to Neo4j and fetching graph data...")
    with driver.session() as session:
        rows = session.execute_read(fetch_subgraph, limit=NEO_BATCH)
    
    print(f"Fetched {len(rows)} relationships. Building visualization...")
    build_pyvis(rows)
    driver.close()

if __name__ == "__main__":
    main()