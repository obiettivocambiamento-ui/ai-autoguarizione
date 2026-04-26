from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import faiss, json, numpy as np, os

app = Flask(__name__)

model = SentenceTransformer("all-MiniLM-L6-v2")

# Se non esiste, crea i dati
if not os.path.exists("site.index"):
    import crawler
    import build_index

index = faiss.read_index("site.index")

with open("chunks.json","r",encoding="utf-8") as f:
    chunks = json.load(f)

memory = {}

def search(query):
    q_emb = model.encode([query])
    D, I = index.search(np.array(q_emb), k=4)
    return [chunks[i] for i in I[0]]

def detect_intent(text):
    text = text.lower()
    if "come" in text:
        return "pratico"
    if "perché" in text:
        return "spiegazione"
    return "generico"

@app.route("/chat", methods=["POST"])
def chat():
    user = request.json["message"]
    user_id = request.json.get("user_id","default")

    if user_id not in memory:
        memory[user_id] = []

    memory[user_id].append(user)

    relevant = search(user)
    context = "\n\n".join(relevant)

    intent = detect_intent(user)

    if intent == "pratico":
        style = "Rispondi con suggerimenti pratici."
    elif intent == "spiegazione":
        style = "Spiega in modo semplice."
    else:
        style = "Rispondi chiaramente."

    reply = f"""
{style}

{context[:500]}

👉 Vuoi approfondire meglio questo tema?
"""

    return jsonify({"reply": reply})

@app.route("/reindex")
def reindex():
    import crawler
    import build_index
    return "Aggiornato"

@app.route("/")
def home():
    return "AI attiva"

app.run(host="0.0.0.0", port=10000)