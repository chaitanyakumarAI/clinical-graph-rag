"""
vector/qdrant_client.py
Stage 1/3 — Qdrant vector store wrapper.

Collection name: medical_chunks
Payload stored per point: chunk_id, text, source, page, chunk_index
"""

from __future__ import annotations

import os
import uuid
from typing import List, Dict

from dotenv import load_dotenv

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medical_chunks")


class QdrantStore:
    def __init__(self, collection: str = COLLECTION_NAME):
        from qdrant_client import QdrantClient

        self.collection = collection
        self.client = QdrantClient(url=QDRANT_URL)
        self._ensure_collection()

    def _ensure_collection(self):
        from qdrant_client.models import Distance, VectorParams
        from ingestion.embedder import get_embedding_dim

        collections = self.client.get_collections().collections
        existing = [c.name for c in collections]

        if self.collection not in existing:
            dim = get_embedding_dim()

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE,
                ),
            )

            print(f"[qdrant] Created collection '{self.collection}' (dim={dim})")

    def upsert_chunks(self, chunks: List[Dict], batch_size: int = 100):
        """
        Store chunk embeddings in Qdrant.
        """
        from qdrant_client.models import PointStruct

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]

            points = []

            for chunk in batch:
                if "embedding" not in chunk:
                    continue

                points.append(
                    PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["chunk_id"])),
                        vector=chunk["embedding"],
                        payload={
                            "chunk_id": chunk["chunk_id"],
                            "text": chunk["text"],
                            "source": chunk["source"],
                            "page": chunk["page"],
                            "chunk_index": chunk["chunk_index"],
                            "token_count": chunk.get("token_count", 0),
                        },
                    )
                )

            if points:
                self.client.upsert(
                    collection_name=self.collection,
                    points=points,
                )

        print(f"[qdrant] ✅ Upserted {len(chunks)} chunks into '{self.collection}'")

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict]:
        """
        Dense vector search.

        Returns:
            List of dictionaries containing chunk_id, text, source, page and score.
        """

        response = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=top_k,
            score_threshold=score_threshold,
        )

        results = response.points

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "text": r.payload["text"],
                "source": r.payload["source"],
                "page": r.payload["page"],
                "score": r.score,
            }
            for r in results
        ]

    def count(self) -> int:
        return self.client.count(
            collection_name=self.collection
        ).count

    def delete_by_source(self, filename: str) -> int:
        """
        Delete all Qdrant points whose payload.source == filename.

        Used when a document is removed from the registry — this purges its
        embeddings from the vector store.

        Returns:
            Approximate number of points deleted (Qdrant does not return an
            exact count from delete operations; we measure before and after).
        """
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        before = self.count()
        self.client.delete(
            collection_name=self.collection,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="source",
                        match=MatchValue(value=filename),
                    )
                ]
            ),
        )
        after = self.count()
        deleted = max(0, before - after)
        print(f"[qdrant] 🗑️  Deleted ~{deleted} points for source='{filename}'")
        return deleted


if __name__ == "__main__":
    store = QdrantStore()
    print(f"Collection '{COLLECTION_NAME}': {store.count()} points")
