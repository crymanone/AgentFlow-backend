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

# --- 1. CONFIGURACIÓN E INICIALIZACIÓN ---

# Configura Gemini (es seguro hacerlo a nivel global)
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("API de Gemini configurada.")
except KeyError:
    print("ADVERTENCIA: API Key de Gemini no encontrada.")

app = FastAPI(title="AgentFlow Production Backend")

# Configura CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# [LA CORRECCIÓN] Función de inicialización "inteligente" de Firebase Admin
def initialize_firebase_admin_once():
    # El SDK de Firebase Admin es inteligente. Si ya existe una app, no la reinicializa.
    # El `if not firebase_admin._apps:` previene que se intente inicializar si ya lo está.
    if not firebase_admin._apps:
        try:
            cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"]
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            print("Firebase Admin SDK inicializado por primera vez para esta instancia.")
        except Exception as e:
            print(f"ERROR CRÍTICO al inicializar Firebase Admin: {e}")
            # En un entorno de producción real, esto debería lanzar una alerta.
            return False
    return True

# --- 2. DECORADOR DE AUTENTICACIÓN ---
def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        initialize_firebase_admin_once() # Se asegura de que Firebase esté vivo
        
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Falta token de autorización.")
        
        id_token = auth_header.split("Bearer ")[1]
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.state.user = decoded_token
        except Exception as e:
            # Capturamos el error específico y lo devolvemos para tener más claridad
            error_detail = f"Token inválido: {e}"
            print(error_detail)
            raise HTTPException(status_code=403, detail=error_detail)
        
        return await f(request, *args, **kwargs)
    return decorated

# --- 3. LÓGICA DE IA Y DATOS (sin cambios) ---

def interpret_intent_with_gemini(text: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest'); prompt = f'Analiza: "{text}". Extrae "action" y "parameters" en JSON.'; response = model.generate_content(prompt); return json.loads(response.text)
def get_emails_for_user(user_id: str) -> list:
    return [{"from": "jefe@empresa.com", "subject": "Revisión presupuesto Q4"}, {"from": "soporte@saas.com", "subject": "Ticket #481516"}]
def summarize_emails_with_gemini(emails: list) -> str:
    model = genai.GenerativeModel('gemini-1.5-pro-latest'); prompt = f'Eres Aura. Resume estos correos de forma ejecutiva: {json.dumps(emails)}'; response = model.generate_content(prompt); return response.text.strip()

# --- 4. ENDPOINTS DE LA API ---

@app.get("/")
def root():
    return {"status": "AgentFlow Backend Activo"}

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]; text = data.get("text", "")
    intent = interpret_intent_with_gemini(text); action = intent.get("action")
    if action == "summarize_inbox":
        try:
            emails = get_emails_for_user(user_id); summary = summarize_emails_with_gemini(emails)
            return {"action": "summarize_inbox", "payload": {"summary": summary}}
        except Exception as e: return {"action": "error", "payload": {"message": f"Error en 'summarize_inbox': {e}"}}
    else: return {"action": "unknown", "payload": {"message": f"Acción '{action}' no implementada."}}

# --- 5. ENDPOINTS DE OAUTH2 ---
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback"

@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token"); scope = "https://www.googleapis.com/auth/gmail.readonly"
    url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={REDIRECT_URI_GOOGLE}"
           f"&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={id_token}")
    return RedirectResponse(url=url)

@app.get("/google/callback")
async def google_callback(request: Request):
    initialize_firebase_admin_once() # Se asegura de que Firebase esté vivo
    db = firestore.client() # Obtenemos una instancia de la BD

    id_token = request.query_params.get("state"); code = request.query_params.get("code")
    try:
        decoded_token = auth.verify_id_token(id_token); user_id = decoded_token["uid"]
        token_url = "https://oauth2.googleapis.com/token"
        data = {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "code": code,
                "redirect_uri": REDIRECT_URI_GOOGLE, "grant_type": "authorization_code"}
        r = requests.post(token_url, data=data); tokens = r.json()
        
        user_ref = db.collection("users").document(user_id)
        user_ref.collection("connected_accounts").document("google").set(tokens)
        
        return JSONResponse(content={"status": "Cuenta de Google conectada. Puedes cerrar esta ventana."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")