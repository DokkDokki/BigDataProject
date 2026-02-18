from fastapi import FastAPI
from ml_utils import calculate_risk_score

app = FastAPI()

@app.post("/get_risk")
async def get_risk(data: dict):
    # This takes data from Node.js and returns the score
    score = calculate_risk_score(data)
    return {"risk_score": score}

# uvicorn main:app --reload