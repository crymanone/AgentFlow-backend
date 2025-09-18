# Este código se guardaría como 'api/index.py' en nuestro repositorio.
# Es el corazón de nuestro backend, sin cambios en su lógica.

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Literal

# --- Modelos de Datos (El contrato no cambia) ---
class ActivityItem(BaseModel):
    id: str
    type: Literal["DRAFT_READY", "CLASSIFICATION", "SUMMARY_DONE"]
    title: str
    subtitle: str
    timestamp: str

class DashboardData(BaseModel):
    agent_name: str
    status_text: str
    is_active: bool
    time_saved_minutes: int
    activity_feed: List[ActivityItem]

# --- La Aplicación FastAPI (Ahora llamada 'app' para Vercel) ---
app = FastAPI()

def get_dashboard_data_from_backend():
    # Los datos siguen siendo simulados, pero ahora listos para ser servidos globalmente.
    return {
        "agent_name": "Alfred",
        "status_text": "🟢 Activo y gestionando tu correo.",
        "is_active": True,
        "time_saved_minutes": 45,
        "activity_feed": [
            {"id": "draft-001", "type": "DRAFT_READY", "title": "Borrador listo para 'Consulta de Precios'.", "subtitle": "De: cliente.potencial@email.com", "timestamp": "Hace 5 min"},
            {"id": "class-002", "type": "CLASSIFICATION", "title": "Correo de 'Marketing SaaS' archivado.", "subtitle": "Filtrado automático.", "timestamp": "Hace 12 min"},
        ]
    }

@app.get("/dashboard", response_model=DashboardData)
async def get_dashboard():
    """Endpoint principal del dashboard."""
    return get_dashboard_data_from_backend()

# Nota: El comando uvicorn ya no es necesario, Vercel gestiona el servidor.
print("Código de 'api/index.py' definido y listo para Vercel.")
