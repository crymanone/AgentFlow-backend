import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Literal

# (Modelos Pydantic no cambian)
class ActivityItem(BaseModel):
    id: str; type: Literal["DRAFT_READY", "CLASSIFICATION", "HIGH_PRIORITY", "SUMMARY_DONE"]; title: str; subtitle: str; timestamp: str
class DashboardData(BaseModel):
    agent_name: str; status_text: str; is_active: bool; time_saved_minutes: int; activity_feed: conlist(ActivityItem, min_length=0)

app = FastAPI()

@app.get("/health")
async def health_check():
    try:
        api_key_status = "Loaded" if os.environ.get("GEMINI_API_KEY") else "Not Found"
        return {"status": "ok", "gemini_api_key_status": api_key_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard():
    activity_feed = []
    gemini_enabled = False
    status_text = "🟠 Modo Simulado"

    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        # [LA ÚNICA LÍNEA CAMBIADA] Usamos el modelo recomendado y más reciente.
        model = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config=generation_config)
        gemini_enabled = True
        status_text = "🟢 Analizando con IA Gemini."
    except KeyError:
        activity_feed = [{"id": "mock-001", "type": "CLASSIFICATION", "title": "API Key no encontrada", "subtitle": "La variable de entorno no está configurada.", "timestamp": "Ahora"}]

    if gemini_enabled:
        emails = [{"from": "test@test.com", "subject": "test", "body": "test"}]
        prompt = f"""
        Convierte esta lista a JSON: {json.dumps(emails)}.
        Tu respuesta debe ser ÚNICA Y EXCLUSIVAMENTE una lista de objetos JSON válida, que se pueda parsear directamente.
        Cada objeto debe tener: "id", "type", "title", "subtitle", "timestamp".
        El tipo debe ser "CLASSIFICATION".
        """
        try:
            response = model.generate_content(prompt)
            activity_feed = json.loads(response.text)
            if not activity_feed:
                 raise ValueError("Gemini returned an empty list.")
        except Exception as e:
            print(f"ERROR en GEMINI: {e}")
            activity_feed = [{"id": "err-ai-001", "type": "HIGH_PRIORITY", "title": "Error al procesar", "subtitle": str(e), "timestamp": "Ahora"}]
    
    return { "agent_name": "Aura ✨", "status_text": status_text, "is_active": True, "time_saved_minutes": 120, "activity_feed": activity_feed }