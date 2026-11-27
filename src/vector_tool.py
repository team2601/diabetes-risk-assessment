import chromadb
from src.config import CHROMA_PATH, COLLECTION_NAME

_collection = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_PATH)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def search_health_docs(query: str, k: int = 3) -> dict:
    if not query or not query.strip():
        return {"status": "error", "message": "Query is empty"}
    try:
        k = max(1, min(int(k), 5))
        res = _get_collection().query(query_texts=[query], n_results=k)
        results = [
            {
                "text": doc,
                "title": meta["title"],
                "source_url": meta["source_url"],
                "distance": round(dist, 3),
            }
            for doc, meta, dist in zip(
                res["documents"][0], res["metadatas"][0], res["distances"][0]
            )
        ]
        if not results:
            return {"status": "no_results", "message": "No relevant passages found"}
        return {"status": "ok", "query": query, "results": results}
    except Exception as e:
        return {"status": "error", "message": f"Vector search failed: {e}"}


if __name__ == "__main__":
    for q in [
        "how to prevent prediabetes",
        "HbA1c test",
        "symptoms of high blood sugar",
    ]:
        r = search_health_docs(q)
        print(q, "->", [x["title"] for x in r.get("results", [])])