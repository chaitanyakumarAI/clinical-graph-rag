"""Quick connectivity test — run this first to verify Neo4j is reachable."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from graph.neo4j_client import Neo4jClient

client = Neo4jClient()
try:
    print(client.verify_connection())
    print("✅ Ready to run bootstrap!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Start Neo4j first, then re-run.")
finally:
    client.close()
