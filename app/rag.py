import os
import uuid
import httpx
from typing import List, Tuple
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
QDRANT_URL = os.getenv("QDRANT_URL", "http://host.docker.internal:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "1c_knowledge_base")
VECTOR_SIZE = 1024  # Фиксированная размерность модели BAAI/bge-m3

qdrant = QdrantClient(url=QDRANT_URL)


def init_qdrant_collection():
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME not in collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


async def get_embedding(text: str) -> List[float]:
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embeddings",
            json={"model": EMBED_MODEL, "prompt": text}
        )
        resp.raise_for_status()
        return resp.json()["embedding"]


async def search_knowledge(query: str, limit: int = 4) -> Tuple[str, List[str]]:
    """Ищет релевантные фрагменты инструкций в Qdrant и возвращает текст контекста + список источников."""
    try:
        query_vector = await get_embedding(query)
        hits = qdrant.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=limit,
            score_threshold=0.45
        )
        if not hits:
            return "", []

        chunks = []
        sources = set()
        for hit in hits:
            payload = hit.payload or {}
            source = payload.get("source", "База знаний 1С")
            text = payload.get("text", "")
            if text:
                chunks.append(f"[Источник: {source}]\n{text}")
                sources.add(source)

        return "\n\n---\n\n".join(chunks), list(sources)
    except Exception as e:
        print(f"[RAG Warning] Ошибка поиска в Qdrant: {e}")
        return "", []


async def index_text_chunks(source_name: str, chunks: List[str]) -> int:
    """Индексирует список текстовых фрагментов в коллекцию Qdrant."""
    init_qdrant_collection()
    points = []
    for idx, chunk in enumerate(chunks):
        if not chunk.strip():
            continue
        vec = await get_embedding(chunk)
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={"source": source_name, "chunk_index": idx, "text": chunk}
            )
        )
    if points:
        qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)
