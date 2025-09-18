import os
import json
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel, conlist
from typing import List, Literal

# (Modelos Pydantic no cambian)
class ActivityItem(BaseModel):
    id: str; type: Literal["DRAFT_READY", "CLASSIFICATION", "HIGH_PRIORITY", "SUMMARY_DONE"]; title: str; subtitle: str; timestamp: str
class DashboardData(BaseModel):
    agent_name: str; status_text: str; is_active: bool; time_saved_minutes: int; activity_feed: conlist(ActivityItem, min_length=0)

app = FastAPI()

def get_mock_emails():
    return [
        {"from": "cliente.curioso@email.com", "subject": "Consulta sobre vuestros planes", "body": "Hola, me gustaría saber más sobre el plan Business."},
        {"from": "ceo@nuestraempresa.com", "subject": "URGENTE: Cifras reunión 15h", "body": "Necesito el informe de ventas del Q3 ASAP."},
        {"from": "newsletter@tech-digest.com", "subject": "Las 5 noticias de IA", "body": "Descubre los últimos avances..."},
    ]

@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard():
    activity_feed = []
    gemini_enabled = False
    status_text = "🟠 Modo Simulado"

    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # [NUEVO] Configuramos el modelo para que la salida sea JSON
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
        model = genai.GenerativeModel('gemini-pro', generation_config=generation_config)
        gemini_enabled = True
        status_text = "🟢 Analizando con IA Gemini."
    except KeyError:
        activity_feed = [{"id": "mock-001", "type": "CLASSIFICATION", "title": "Modo de respaldo", "subtitle": "La API de IA no está configurada.", "timestamp": "Ahora"}]

    if gemini_enabled:
        emails = get_mock_emails()
        # [NUEVO] Prompt ultra-específico. La clave es el final.
        prompt = f"""
        Analiza la siguiente lista de emails y conviértela a una lista de objetos JSON.
        Emails: {json.dumps(emails)}
        La salida debe ser ÚNICA Y EXCLUSIVAMENTE una lista de objetos JSON válida, que se pueda parsear directamente.
        No incluyas texto explicativo, ni introducciones, ni las palabras "json" o markdown ```.
        Tu respuesta debe empezar con "[" y terminar con "]".
        El formato de cada objeto debe ser: "id", "type" ("DRAFT_READY", "CLASSIFICATION", "HIGH_PRIORITY"), "title", "subtitle", y "timestamp".
        """
        try:
            response = model.generate_content(prompt)
            # Con la nueva configuración, la respuesta debería ser JSON puro.
            activity_feed = json.loads(response.text)
        except Exception as e:
            print(f"ERROR al procesar la respuesta de GEMINI: {e}")
            activity_feed = [{"id": "err-ai-001", "type": "HIGH_PRIORITY", "title": "Error al procesar con IA", "subtitle": "La respuesta de Gemini no fue válida.", "timestamp": "Ahora"}]

    return { "agent_name": "Aura ✨", "status_text": status_text, "is_active": True, "time_saved_minutes": 120, "activity_feed": activity_feed }