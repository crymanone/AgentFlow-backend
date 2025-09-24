import os
import json
from functools import wraps
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

# --- SDKs de Google y Firebase ---
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- 1. INICIALIZACIÓN DE SERVICIOS ---

# Inicializa Firebase Admin SDK
try:
    cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"]
    cred_dict = json.loads(cred_json_str)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK inicializado correctamente.")
except (KeyError, json.JSONDecodeError):
    db = None
    print("ADVERTENCIA: Credenciales de Firebase Admin no encontradas o corruptas.")

# Configura Gemini
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("API de Gemini configurada.")
except KeyError:
    print("ADVERTENCIA: API Key de Gemini no encontrada.")

app = FastAPI(title="AgentFlow Production Backend")

# --- 2. CONFIGURACIÓN DE CORS ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. DECORADOR DE AUTENTICACIÓN ---
def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Falta token de autorización.")
        id_token = auth_header.split("Bearer ")[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.state.user = decoded_token
        except Exception as e:
            raise HTTPException(status_code=403, detail=f"Token inválido: {e}")
        return await f(request, *args, **kwargs)
    return decorated

# --- 4. LÓGICA DE IA Y DATOS ---

def interpret_intent_with_gemini(text: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Analiza este comando: '{text}'.
    Extrae la 'action' (posibles: "summarize_inbox", "search_emails", "create_draft") y los 'parameters'.
    Responde ÚNICAMENTE con un JSON válido.
    Ej: si el comando es "resume correos de acme de ayer", responde {{"action": "summarize_inbox", "parameters": {{"client_name": "acme", "time_period": "yesterday"}}}}
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        print(f"Error interpretando intención con Gemini: {e}")
        return {"action": "error", "parameters": {"message": "No se pudo interpretar el comando."}}

def get_emails_for_user(user_id: str) -> list:
    print(f"Buscando emails para el usuario (simulado): {user_id}...")
    return [
        {"from": "jefe@empresa.com", "subject": "Revisión urgente del presupuesto Q4", "snippet": "Necesito que le eches un vistazo antes de la reunión de las 3pm."},
        {"from": "marketing@newsletter.com", "subject": "Novedades de la semana en IA", "snippet": "Descubre las últimas tendencias..."},
        {"from": "soporte@saas.com", "subject": "Ticket #481516: Consulta sobre facturación", "snippet": "Un cliente tiene una duda sobre su factura."},
    ]

def summarize_emails_with_gemini(emails: list) -> str:
    prompt = f"""
    Eres Aura, una asistente de IA. Genera un resumen ejecutivo de estos correos.
    Destaca lo más importante y las acciones requeridas. Sé concisa y profesional.
    Correos: {json.dumps(emails)}
    Responde solo con el texto del resumen.
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text.strip()

# --- 5. ENDPOINTS DE LA API ---

@app.get("/")
def root():
    return {"status": "AgentFlow Backend Activo"}

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]
    text = data.get("text", "")
    intent = interpret_intent_with_gemini(text)
    action = intent.get("action")
    parameters = intent.get("parameters", {})
    
    if action == "summarize_inbox":
        try:
            emails = get_emails_for_user(user_id)
            summary = summarize_emails_with_gemini(emails)
            return {"action": "summarize_inbox", "payload": {"summary": summary}}
        except Exception as e:
            return {"action": "error", "payload": {"message": f"Error en 'summarize_inbox': {e}"}}
    elif action == "error":
        return {"action": "error", "payload": parameters}
    else:
        return {"action": "unknown", "payload": {"message": f"Acción '{action}' reconocida pero no implementada."}}

# --- 6. ENDPOINTS DE OAUTH2 ---
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback"

@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Falta el token de usuario.")
    
    scope = "https://www.googleapis.com/auth/gmail.readonly"
    url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI_GOOGLE}&response_type=code&scope={scope}"
        f"&access_type=offline&prompt=consent&state={id_token}"
    )
    return RedirectResponse(url=url)

@app.get("/google/callback")
async def google_callback(request: Request):
    id_token = request.query_params.get("state")
    code = request.query_params.get("code")
    if not id_token or not code:
        raise HTTPException(status_code=400, detail="Falta información en el callback.")

    try:
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token["uid"]
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code, "redirect_uri": REDIRECT_URI_GOOGLE, "grant_type": "authorization_code"
        }
        r = requests.post(token_url, data=data)
        tokens = r.json()
        
        if db:
            user_ref = db.collection("users").document(user_id)
            user_ref.collection("connected_accounts").document("google").set(tokens)
        return JSONResponse(content={"status": "Cuenta de Google conectada. Puedes cerrar esta ventana."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")