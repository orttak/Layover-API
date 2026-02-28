import json
import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from google.genai import types

app = FastAPI(title="Layover Architect API")

# --- KONFİGÜRASYON ---
# .env dosyasında saklaman önerilir
client = genai.Client(api_key=GOOGLE_GEMINI_API_KEY, http_options={'api_version': 'v1alpha'})

# --- MODELLER (Request/Response Şemaları) ---
class UserAnswer(BaseModel):
    field: str
    selected_label: str

class PlanRequest(BaseModel):
    city: str
    arrival_utc: str
    departure_utc: str
    weather: str
    frontend_answers: List[UserAnswer]

# --- CORE LOGIC ---
async def call_gemini_architect(city, arrival, departure, weather, answers):
    # Senin mükemmel çalışan promptun
    prompt = f"""
    Role: You are the Layover Architect. 
    Goal: Create a high-precision, JSON-formatted travel itinerary with costs and categories.
    CONTEXT:
    - City: {city} | Weather: {weather}
    - Arrival (UTC): {arrival} | Departure (UTC): {departure}
    - User Preferences: {json.dumps([a.dict() for a in answers])}

    STRICT LOGICAL RULES:
    1. Safe Window: Calculate buffers (30m deplaning, 35m passport, 90m return). 
    2. Categories: Each activity MUST have a 'category' field: "food", "activity", "transport", or "culture".
    3. Costs: Each activity MUST have an 'estimated_cost' string (e.g., "€15", "Free", "€25-40").
    4. Language: English only.
    5. Clean Output: No citations [1], [2]. Plain text only.

    OUTPUT JSON STRUCTURE MUST MATCH THE SPECIFIED SCHEMA.
    """

    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_maps=types.GoogleMaps())],
                temperature=0.1
            )
        )
        
        # Markdown temizliği ve JSON parse
        clean_text = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini Error: {str(e)}")

# --- ENDPOINT ---
@app.post("/test-maps-grounding")
async def generate_plan(request: PlanRequest):
    """
    Kullanıcı tercihlerine göre 6-8 saatlik layover planı üretir.
    """
    plan = await call_gemini_architect(
        request.city, 
        request.arrival_utc, 
        request.departure_utc, 
        request.weather, 
        request.frontend_answers
    )
    
    return {
        "status": "success",
        "model": "gemini-2.5-flash",
        "data": plan
    }

# Çalıştırmak için: uvicorn main:app --reload