from fastapi import FastAPI, Request
import joblib
import pandas as pd
from model import SoccerMatchmakerAI 

app = FastAPI(title="Soccer Matchmaker AI API")

ai_system = joblib.load('matchmaker_ai.pkl')

@app.post("/generate-match")
async def create_match(request: Request, team_size: int = 5):
    try:
        backend_payload = await request.json()
        
        match_result = ai_system.generate_match(backend_payload, team_size=team_size)
        
        return {"status": "success", "data": match_result}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}