import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from graph.neo4j_client import Neo4jClient
from graph.document_registry import get_source_map

if __name__ == "__main__":
    client = Neo4jClient()
    try:
        mapping = get_source_map(client)
        print("Source Map:", mapping)
    except Exception as e:
        print("Error:", e)
