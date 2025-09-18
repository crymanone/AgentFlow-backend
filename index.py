import os
import json
import google.generativeai as genai
from fastapi import FastAPI
from pydantic import BaseModel, Field, conlist
from typing import List, Literal

# --- 1. CONFIGURACIÓN DE IA ---
# Leemos la clave de API de forma segura desde las variables de entorno de Vercel
try:
    api_key = os.environ["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-pro')
    GEMINI_ENABLED = True
except KeyError:
    print("ADVERTENCIA: La variable de entorno GEMINI_API_KEY no se encontró.")
    GEMINI_ENABLED = False

# --- 2. MODELOS DE DATOS (EL CONTRATO CON LA APP) ---
# Se mantiene igual, pero añadimos validación para la lista
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
    activity_feed: conlist(ActivityItem, min_length=0) # Asegura que sea una lista de ActivityItems

# --- 3. LA APLICACIÓN FASTAPI ---
app = FastAPI()

# --- 4. SIMULACIÓN DE DATOS DE ENTRADA (PRONTO SERÁ LA API DE GMAIL) ---
def get_mock_emails():
    return [
        {"from": "cliente.curioso@email.com", "subject": "Consulta sobre vuestros planes", "body": "Hola, me gustaría saber más sobre el plan Business. ¿Qué incluye exactamente?"},
        {"from": "ceo@nuestraempresa.com", "subject": "URGENTE: Cifras para la reunión de las 15h", "body": "Necesito el informe de ventas del Q3 lo antes posible para la reunión con los inversores."},
        {"from": "newsletter@tech-digest.com", "subject": "Las 5 noticias de IA que no te puedes perder esta semana", "body": "Descubre los últimos avances en modelos de lenguaje..."},
    ]

# --- 5. EL CEREBRO DE AURA (PROMPT ENGINEERING CON GEMINI) ---
def analyze_emails_with_gemini(emails: List[dict]) -> List[dict]:
    # El prompt es la "personalidad y las instrucciones" de Aura
    prompt = f"""
    Eres Aura, una asistente de IA superdotada. Tu misión es analizar una lista de emails y transformarla en un feed de actividad para un dashboard. La respuesta DEBE ser un JSON válido que contenga una lista de objetos. No incluyas nada antes o después del JSON.

    Emails a analizar:
    {json.dumps(emails)}

    Genera una lista JSON para el "activity_feed". Cada objeto debe tener los siguientes campos: "id" (un string único como "draft-001"), "type" (puede ser "DRAFT_READY", "CLASSIFICATION", "HIGH_PRIORITY", o "SUMMARY_DONE"), "title" (un título corto y descriptivo), "subtitle" (con información clave como el remitente), y "timestamp" (un tiempo relativo como "Hace 5 min").

    - Para consultas de clientes, usa el tipo "DRAFT_READY".
    - para emails urgentes de personas importantes, usa "HIGH_PRIORITY".
    - Para newsletters y notificaciones, usa "CLASSIFICATION" y un título que indique archivado o clasificado.
    """
    try:
        response = model.generate_content(prompt)
        # Limpiamos la respuesta para asegurar que es solo JSON
        clean_json_response = response.text.strip().replace("```json", "").replace("```", "")
        return json.loads(clean_json_response)
    except Exception as e:
        print(f"Error al procesar la respuesta de Gemini: {e}")
        # Si la IA falla, devolvemos un mensaje de error claro en el feed
        return [{
            "id": "err-ai-001",
            "type": "HIGH_PRIORITY",
            "title": "Error al contactar con la IA",
            "subtitle": "No se pudo generar el feed de actividad.",
            "timestamp": "Ahora"
        }]

# --- 6. FUNCIÓN DE RESPALDO (SI GEMINI ESTÁ DESACTIVADO) ---
def get_mock_activity_feed():
    return [
        {"id": "mock-001", "type": "CLASSIFICATION", "title": "Modo de respaldo activo", "subtitle": "La API de IA no está configurada.", "timestamp": "Ahora"}
    ]

# --- 7. EL ENDPOINT PRINCIPAL QUE LO UNE TODO ---
@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard():
    emails = get_mock_emails()
    
    if GEMINI_ENABLED:
        activity_feed = analyze_emails_with_gemini(emails)
    else:
        activity_feed = get_mock_activity_feed()

    return {
        "agent_name": "Aura ✨", # Le damos un nombre más acorde
        "status_text": "🟢 Analizando con IA Gemini." if GEMINI_ENABLED else "🟠 Modo Simulado",
        "is_active": True,
        "time_saved_minutes": 120, # ¡La IA nos ahorra más tiempo!
        "activity_feed": activity_feed
    }