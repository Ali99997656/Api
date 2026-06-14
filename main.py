from fastapi import FastAPI, Request
import joblib
import pandas as pd
from model import SoccerMatchmakerAI 

app = FastAPI(title="Soccer Matchmaker AI API")

ai_system = joblib.load('matchmaker_ai.pkl')

# 2. إنشاء المسار (Endpoint) اللي رح يتواصل معه الباك إند
@app.post("/generate-match")
async def create_match(request: Request, team_size: int = 5):
    try:
        # استلام البيانات الخام (JSON) من الباك إند
        backend_payload = await request.json()
        
        # تمرير البيانات للموديل لاستخراج التشكيلة
        match_result = ai_system.generate_match(backend_payload, team_size=team_size)
        
        # إرجاع النتيجة للباك إند
        return {"status": "success", "data": match_result}
        
    except Exception as e:
        return {"status": "error", "message": str(e)}