import os
import json
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist
from typing import List, Literal, Optional

# --- 1. MODELOS DE DATOS ---

# Modelos para /dashboard
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

# [NUEVO] Modelos para /api/voice-command
class VoiceCommandInput(BaseModel):
    text: str

class ActionParameter(BaseModel):
    client_name: Optional[str] = None
    time_period: Optional[str] = None
    subject_keywords: Optional[List[str]] = None
    error_message: Optional[str] = None

class VoiceCommandOutput(BaseModel):
    action: str
    parameters: ActionParameter

# --- 2. LA APLICACIÓN FASTAPI ---
app = FastAPI()

# --- 3. ENDPOINTS DE LA API ---

@app.get("/health")
async def health_check():
    try:
        api_key_status = "Loaded" if os.environ.get("GEMINI_API_KEY") else "Not Found"
        return {"status": "ok", "gemini_api_key_status": api_key_status}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard():
    # (Esta función no ha cambiado desde la v2.5)
    # ... (código idéntico a la versión anterior)
    activity_feed = []
    gemini_enabled = False
    status_text = "🟠 Modo Simulado"

    try:
        api_key = os.environ["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        generation_config = genai.types.GenerationConfig(response_mime_type="application/json")
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

# --- [NUEVO] El Cerebro Lingüístico de Aura ---
def parse_command_with_gemini(text: str) -> dict:
    prompt = f"""
    Eres el "Motor de Comprensión de Lenguaje Natural" de la IA Aura.
    Tu tarea es analizar la transcripción de un comando de voz y convertirlo a una acción JSON estructurada.
    Transcripción del Usuario: "{text}"
    Analiza la transcripción para identificar la "action" principal y sus "parameters".
    Las acciones posibles son: "summarize_inbox", "search_emails", "draft_reply", "error".
    Los parámetros posibles son: "client_name", "time_period", "subject_keywords".
    Ejemplos:
    - texto: "resume mi bandeja de entrada" -> acción: "summarize_inbox"
    - texto: "busca los correos de TechCorp del último mes" -> acción: "search_emails", parámetros: {{"client_name": "TechCorp", "time_period": "last_30_days"}}
    Basado en el texto proporcionado, genera el objeto JSON correspondiente.
    Tu respuesta debe ser ÚNICA Y EXCLUSIVAMENTE un objeto JSON válido, que empiece con '{{' y termine con '}}'.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"ERROR en el parseo del comando de voz: {e}")
        return {"action": "error", "parameters": {"error_message": str(e)}}

# --- [NUEVO] El Endpoint para Comandos de Voz ---
@app.post("/api/voice-command", response_model=VoiceCommandOutput)
async def handle_voice_command(command: VoiceCommandInput):
    """
    Recibe una transcripción de voz, la procesa con la IA
    y devuelve una acción estructurada.
    """
    action_json = parse_command_with_gemini(command.text)
    return action_json