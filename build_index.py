from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")

with open("chunks.json","r",encoding="utf-8") as f:
    chunks = json.load(f)

embeddings = model.encode(chunks)

dimension = embeddings[0].shape[0]
index = faiss.IndexFlatL2(dimension)

index.add(np.array(embeddings))

faiss.write_index(index, "site.index")