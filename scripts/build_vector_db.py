import glob
import os
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.config import CHROMA_PATH, COLLECTION_NAME

client = chromadb.PersistentClient(path=CHROMA_PATH)
try:
    client.delete_collection(COLLECTION_NAME)
except Exception:
    pass  # first run: nothing to delete

collection = client.create_collection(COLLECTION_NAME)
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)

ids, texts, metas = [], [], []
for path in sorted(glob.glob("docs/*.txt")):
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    title, url, body = lines[0].strip(), lines[1].strip(), "\n".join(lines[2:])
    topic = os.path.splitext(os.path.basename(path))[0]
    for i, chunk in enumerate(splitter.split_text(body)):
        ids.append(f"{topic}-{i}")
        texts.append(chunk)
        metas.append({"title": title, "source_url": url, "topic": topic})

collection.add(ids=ids, documents=texts, metadatas=metas)
print(f"Stored {len(ids)} chunks from {len(set(m['topic'] for m in metas))} files")