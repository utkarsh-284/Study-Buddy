import json
import asyncio
from typing import TypedDict, List, Dict, Any
from functools import lru_cache
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from neo4j import AsyncGraphDatabase, GraphDatabase
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate

import config


# -----------------------------
# Config
# -----------------------------
EMBED_MODEL = "nvidia/llama-3.2-nemoretriever-300m-embed-v2"
CHAT_MODEL = "meta/llama-3.3-70b-instruct"
TOP_K = 5

INDEX_NAME = config.PINECONE_INDEX_NAME


# -----------------------------
# Initialize clients
# -----------------------------
client = OpenAI(
  api_key=config.NVIDIA_API_KEY,
  base_url="https://integrate.api.nvidia.com/v1"
)
pc = Pinecone(api_key=config.PINECONE_API_KEY)

# Connect to Pinecone index
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

index = pc.Index(INDEX_NAME)

# Connect to Neo4j
driver = AsyncGraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)


### ===== CACHED EMBEDDINGS =====

@lru_cache(maxsize=128)   # IMPROVEMENT 1: Caching embeddings
def embed_text(text: str) -> List[float]:
    """Get embedding for a text string with caching."""
    resp = client.embeddings.create(
        model=EMBED_MODEL, 
        input=[text],
        extra_body={"input_type": "query", "truncate": "NONE"})   # for the type of embedding model

    return resp.data[0].embedding


### ===== LANGGRAPH AGENT STATE AND NODES =====

# Defining the State for our graph
class RAGState(TypedDict):
    query: str
    vector_matches: List[Dict]
    summary: str
    graph_facts: List[Dict]
    generation: str


# Node 1: Retrive from Pinecone
def retrieve_vector_context(state: RAGState) -> RAGState:
    """Query Pinecone for semantic matches."""
    print("--- Step 1: Retrieving vector context ---")
    vec = embed_text(state['query'])
    res = index.query(
        vector=vec,
        top_k=TOP_K,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Found {len(res['matches'])} vector matches.")
    return {"vector_matches": res['matches']}


# Node 2: Summarizing
# Node 2: Summarizing
def summarize_retrieved_context(state: RAGState) -> Dict[str, str]:
    """Summarize the initial retrieved nodes to distill the key themes."""
    print("--- Step 2: Summarizing retrieved context ---")

    # <-- MODIFIED: Use text_snippet from your new vector schema -->
    retrieved_snippets = [m['metadata'].get('text_snippet', '') for m in state['vector_matches']]

    # <-- MODIFIED: New prompt for a study bot -->
    summary_prompt = f"""
    Based on the following text snippets from a document, please provide a very brief, one-paragraph summary.
    Identify the main academic concepts, people, or topics that seem most relevant to the user's original query: "{state['query']}".

    This summary will be used to find related ideas in a knowledge graph.

    Retrieved Snippets:
    {retrieved_snippets}
    """

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[{'role': 'user', 'content': summary_prompt}],
        max_tokens=250, # Increased token limit for better summary
        temperature=0.1
    )
    summary = resp.choices[0].message.content
    print("DEBUG: Generated summary of top nodes.")
    return {'summary': summary}


# Node 3: Retrieve from Neo4j
async def retrieve_graph_context(state: RAGState) -> Dict[str, Any]:
    """Fetch neighbouring nodes from Neo4j based on vector matches."""
    print("---Step 3: Retrieving graph context ---")
    
    # The vector matches give us the :Chunk node IDs
    chunk_node_ids = [m['id'] for m in state['vector_matches']]
    
    async def fetch_facts_for_id(nid):
        async with driver.session() as session:
            
            # <-- This query is now specific to your StudyBuddy graph -->
            q = (
                # 1. Match the :Chunk node using its ID
                "MATCH (c:StudyBuddy:Chunk {id: $nid}) "
                # 2. Find any entity it mentions or defines
                "MATCH (c)-[r:MENTIONS|DEFINES]->(m:StudyBuddy) "
                # 3. Also find concepts related to that entity
                "OPTIONAL MATCH (m)-[r2]-(m2:StudyBuddy) "
                # 4. Ensure we don't just get other chunks
                "WHERE NOT 'Chunk' IN labels(m) AND NOT 'Chunk' IN labels(m2) "
                # 5. Return the facts
                "RETURN "
                "  m.name AS entity_name, "
                "  labels(m)[0] AS entity_label, "
                "  type(r2) AS rel, "
                "  m2.name AS related_entity_name "
                "LIMIT 5" # Limit to 5 facts per chunk
            )
            result = await session.run(q, nid=nid)
            records = await result.data()
            return [
                {
                    "source_chunk": nid,
                    "entity": r["entity_name"],
                    "entity_label": r["entity_label"],
                    "relationship": r["rel"],
                    "related_entity": r["related_entity_name"]
                } for r in records
            ]
        
    tasks = [fetch_facts_for_id(nid) for nid in chunk_node_ids]
    results_list = await asyncio.gather(*tasks)

    facts = [item for sublist in results_list for item in sublist]

    print(f"DEBUG: Found {len(facts)} graph facts.")
    return {"graph_facts": facts}


# Node 4: Generate the final response
def generate_response(state: RAGState) -> RAGState:
    """Generate a response using LLM with both contexts."""
    print("---Step 4: Generate final response ---")

    # <-- MODIFIED: New System Prompt for the Study Buddy -->
    system_prompt = """
    You are a friendly and expert learning assistant. Your goal is to teach a student about a topic in a clear, concise, and lucid way.
    
    Follow these steps to generate your response:
    1.  **Analyze the Goal:** First, understand the user's core question.
    2.  **Synthesize Context:** Review the 'Retrieved Text Snippets' for direct answers and the 'Connected Graph Facts' for related concepts and context.
    3.  **Construct the Answer:** Build a comprehensive answer. Start by directly answering the user's question. Then, use the graph facts to explain *how* this concept relates to other ideas.
    4.  **Cite Sources:** You MUST cite your sources.
        - For information from a text snippet, cite its chunk ID (e.g., `...chunk_1`).
        - For information from the graph, cite the concept (e.g., `(Transformer)`).
    5.  **Be Helpful:** Conclude with 1-2 follow-up questions or related topics the user might want to explore.
    """

    # <-- MODIFIED: Context strings match our new data schema -->
    vec_context_str = "\n".join([
        f"- Chunk ID: {m['id']} (Page {m['metadata'].get('page_number', 0)}): \"{m['metadata'].get('text_snippet', '')}...\"" 
        for m in state['vector_matches']
    ])
    
    graph_context_str = "\n".join([
        f"- ({f['entity']} {f['entity_label']}) is {f['relationship']} ({f['related_entity']})" 
        for f in state['graph_facts'] if f['relationship']
    ])
    # Filter for facts that have a relationship

    user_prompt = f"""
    User Query: {state['query']}

    An initial analysis of the most relevant text snippets produced this summary:
    "{state['summary']}"

    Use this summary, along with the detailed context below, to construct your answer.

    Retrieved Text Snippets:
    {vec_context_str}

    Connected Graph Facts:
    {graph_context_str}

    Now, generate the response using the system instructions. Be clear, helpful, and cite your sources.
    """

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    resp = client.chat.completions.create(
        model = CHAT_MODEL,
        messages=prompt,
        max_tokens=1024, # Increased token limit for detailed explanations
        temperature=0.3
    )
    generation = resp.choices[0].message.content
    return {"generation": generation}


# ===== MAIN CHAT APPLICATION =====

async def main():
    # Define the graph workflow
    workflow = StateGraph(RAGState)
    workflow.add_node("vector_retriever", retrieve_vector_context)
    workflow.add_node("summarizer", summarize_retrieved_context)
    workflow.add_node("graph_retriever", retrieve_graph_context)
    workflow.add_node("generator", generate_response)

    # Define the graph edges
    workflow.set_entry_point("vector_retriever")
    workflow.add_edge("vector_retriever", "summarizer")
    workflow.add_edge("summarizer", "graph_retriever")
    workflow.add_edge("graph_retriever", "generator")
    workflow.add_edge("generator", END)

    # Compile the graph into runnable app
    app = workflow.compile()

    print("Hybrid travel assistant (Async LangGraph Eddition). Type 'exit' to quit.")
    while True:
        query = input('\n Enter your travel question: ').strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        
        # Create the initial input
        initial_input = {"query": query}

        # Run the graph
        final_state = await app.ainvoke(initial_input)

        print("\n ===== ASSISTANT ANSWER =====\n")
        print(final_state['generation'])
        print("\n ===== END ===== \n")

if __name__ == "__main__":
    asyncio.run(main())