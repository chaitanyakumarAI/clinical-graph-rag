"""Quick connectivity test — run this first to verify Neo4j is reachable."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from backend.graph.neo4j_client import Neo4jClient

client = Neo4jClient()
try:
    print(client.verify_connection())
    print("✅ Ready to run bootstrap!")
except Exception as e:
    print(f"❌ Connection failed: {e}")
    print("Start Neo4j first, then re-run.")
finally:
    client.close()
