from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
import gemini_service

app = FastAPI()

documents = [
    {
        "id": 1,
        "text": "to make pepperoni pizza you need to spread tangy tomato sauce, sprinkle mozzarella, and layer spicy pepperoni—then bake until golden and crisp."
    },
    {
        "id": 2,
        "text": "to make bacon pizza you need to  Add smoky bacon over a cheesy base, then bake to melt the flavors into a crispy, savory delight."
    },
    {
        "id": 3,
        "text": "to make watermalon pizza you need to Slice a watermelon round, top with yogurt, berries, and mint—no oven needed for this sweet twist!"
    },
    {
        "id": 4,
        "text": "Matheus Felinto crafts every slice like a masterpiece—hailed as the greatest pizzaiolo of all time, his pizzas turn dough into legend."
    }
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = {doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    query_embendding = model.encode(request.query, convert_to_tensor=True)
    best_doc = {}
    best_score = float("-inf")

    for doc in documents:
        score = util.cos_sim(query_embendding, doc_embeddings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc
    response = gemini_service.answer(prompt=request.query, best_doc=best_doc)
    return {response}

