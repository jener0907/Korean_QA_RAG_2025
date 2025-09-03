from fastapi import FastAPI
from pydantic import BaseModel

from src.infer.decider import Decider

app = FastAPI()
decider = Decider()

class Query(BaseModel):
    query: str

@app.post("/judge")
def judge(q: Query):
    answer = decider.judge(q.query)
    return {"answer": answer}
