import os
import json
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List, Literal

# ... (Los modelos Pydantic no cambian) ...

class ActivityItem(BaseModel):
    id: str
    type: Literal["DRAFT_READY", "CLASSIFICATION", "HIGH_PRIORITY", "SUMMARY_DONE"]
    title: str
    subtitle: str
    timestamp: str

class DashboardData(BaseModel):
    agent_name: str
    status_text: str
    is_active: bool
    time_saved_minutes: int
    activity_feed: conlist(ActivityItem, min_length=0)

app = FastAPI()

def get_mock_emails():
    # ... (no cambia)
    return [
        {"from": "cliente.curioso@email.com", "subject": "Consulta sobre vuestros planes", "body": "Hola, me gustaría saber más sobre el plan Business. ¿Qué incluye exactamente?"},
        {"from": "ceo@nuestraempresa.com", "subject": "URGENTE: Cifras para la reunión de las 15h", "body": "Necesito el informe de ventas del Q3 lo antes posible para la reunión con los inversores."},
        {"from": "newsletter@tech-digest.com", "subject": "Las 5 noticias de IA que no te puedes perder esta semana", "body": "Descubre los últimos avances en modelos de lenguaje..."},
    ]

# --- EL ENDPOINT PRINCIPAL (CON TODA LA LÓGICA DENTRO) ---
@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard():
    
    # [NUEVO] Movemos toda la lógica de la IA aquí dentro
    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-pro')
        GEMINI_ENABLED = True
    except KeyError:
        GEMINI_ENABLED = False

    emails = get_mock_emails()
    activity_feed = []
    
    if GEMINI_ENABLED:
        prompt = f"""
        Eres Aura, una asistente de IA. Tu misión es analizar una lista de emails y transformarla en un feed de actividad para un dashboard. La respuesta DEBE ser un JSON válido que contenga una lista de objetos. No incluyas nada más.

        Emails a analizar:
        {json.dumps(emails)}

        Genera una lista JSON para el "activity_feed". Cada objeto debe tener: "id", "type" ("DRAFT_READY", "CLASSIFICATION", "HIGH_PRIORITY"), "title", "subtitle", y "timestamp".
        """
        try:
            response = model.generate_content(prompt)
            clean_json_response = response.text.strip().replace("```json", "").replace("```", "")
            activity_feed = json.loads(clean_json_response)
        except Exception as e:
            # Si la IA falla, ahora sí lo registraremos de forma más visible
            print(f"ERROR en la llamada a GEMINI: {e}")
            activity_feed = [{"id": "err-ai-001", "type": "HIGH_PRIORITY", "title": "Error al procesar con IA", "subtitle": "La respuesta de Gemini no fue válida.", "timestamp": "Ahora"}]
    else:
        activity_feed = [{"id": "mock-001", "type": "CLASSIFICATION", "title": "Modo de respaldo", "subtitle": "La API de IA no está configurada.", "timestamp": "Ahora"}]

    return {
        "agent_name": "Aura ✨",
        "status_text": "🟢 Analizando con IA Gemini." if GEMINI_ENABLED else "🟠 Modo Simulado",
        "is_active": True,
        "time_saved_minutes": 120,
        "activity_feed": activity_feed
    }