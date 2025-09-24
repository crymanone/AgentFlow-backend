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
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request as GoogleAuthRequest

# --- 1. INICIALIZACIÓN DE SERVICIOS ---
db = None
def initialize_firebase_admin_once():
    global db
    if not firebase_admin._apps:
        try:
            cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"]
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase Admin SDK inicializado.")
        except Exception as e: print(f"ERROR CRÍTICO inicializando Firebase Admin: {e}")

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError: print("ADVERTENCIA: API Key de Gemini no encontrada.")

app = FastAPI(title="AgentFlow Production Backend v4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 2. DECORADOR DE AUTENTICACIÓN ---
def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        initialize_firebase_admin_once()
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "): raise HTTPException(status_code=401, detail="Falta token.")
        id_token = auth_header.split("Bearer ")[1]
        try:
            request.state.user = auth.verify_id_token(id_token)
        except Exception as e: raise HTTPException(status_code=403, detail=f"Token inválido: {e}")
        return await f(request, *args, **kwargs)
    return decorated

# --- 3. LÓGICA DE IA Y DATOS ---
def interpret_intent_with_gemini(text: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Tu tarea es actuar como un router de intenciones. Analiza el comando del usuario y conviértelo a un formato JSON estricto.
    Comando del usuario: "{text}"
    Las acciones válidas son "summarize_inbox", "search_emails", o "unknown".
    Los parámetros válidos son "client_name" (string) y "time_period" (string).
    Ejemplos de conversión:
    - "resume mis correos" -> {{"action": "summarize_inbox", "parameters": {{}}}}
    - "busca los emails de acme de la semana pasada" -> {{"action": "search_emails", "parameters": {{"client_name": "acme", "time_period": "last week"}}}}
    - "cuéntame un chiste" -> {{"action": "unknown", "parameters": {{}}}}
    Proporciona ÚNICA Y EXCLUSIVAMENTE el objeto JSON correspondiente. No añadas texto extra ni explicaciones.
    """
    try:
        response = model.generate_content(prompt)
        print(f"Respuesta cruda de Gemini (Intención): {response.text}") 
        return json.loads(response.text)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error de parseo JSON desde Gemini: {e}")
        return {"action": "error", "parameters": {"message": "La IA devolvió un formato inesperado."}}
    except Exception as e:
        print(f"Error general en Gemini: {e}")
        return {"action": "error", "parameters": {"message": "No se pudo comunicar con la IA."}}

def get_real_emails_for_user(user_id: str) -> list:
    print(f"Buscando correos REALES para el usuario {user_id}...")
    if not db: raise Exception("Base de datos no disponible.")
    
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get()
    if not doc.exists: raise Exception("El usuario no tiene una cuenta de Google conectada.")
    
    tokens = doc.to_dict()
    creds = Credentials(
        token=tokens.get("access_token"), refresh_token=tokens.get("refresh_token"),
        token_uri="https://oauth2.googleapis.com/token", client_id=os.environ.get("GOOGLE_CLIENT_ID"),
        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"), scopes=["https://www.googleapis.com/auth/gmail.readonly"]
    )

    if not creds.valid and creds.expired and creds.refresh_token:
        print("Refrescando token de acceso de Google...")
        creds.refresh(GoogleAuthRequest())
        doc_ref.update({"access_token": creds.token})

    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=5).execute()
        messages = results.get('messages', [])
        emails_list = []
        if messages:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id'], format='metadata', metadataHeaders=['From', 'Subject']).execute()
                headers = msg.get('payload', {}).get('headers', [])
                subject = next((i['value'] for i in headers if i['name'] == 'Subject'), None)
                sender = next((i['value'] for i in headers if i['name'] == 'From'), None)
                emails_list.append({"from": sender, "subject": subject, "snippet": msg.get('snippet', '')})
        return emails_list
    except HttpError as error:
        print(f"Ocurrió un error con la API de Gmail: {error}")
        raise Exception(f"Error de API de Gmail: {error.reason}")

def summarize_emails_with_gemini(emails: list) -> str:
    prompt = f'Eres Aura, una IA asistente. Genera un resumen ejecutivo de estos correos: {json.dumps(emails)}. Sé concisa y profesional.'
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text.strip()

# --- 4. ENDPOINTS DE LA API ---
@app.get("/")
def root(): return {"status": "AgentFlow Backend Activo"}

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]; text = data.get("text", "")
    intent = interpret_intent_with_gemini(text); action = intent.get("action")
    parameters = intent.get("parameters", {})
    
    if action == "summarize_inbox":
        try:
            emails = get_real_emails_for_user(user_id)
            if not emails:
                return {"action": "summarize_inbox", "payload": {"summary": "No tienes correos nuevos en tu bandeja de entrada."}}
            summary = summarize_emails_with_gemini(emails)
            return {"action": "summarize_inbox", "payload": {"summary": summary}}
        except Exception as e:
            return {"action": "error", "payload": {"message": str(e)}}
    elif action == "error": return {"action": "error", "payload": parameters}
    else: return {"action": "unknown", "payload": {"message": f"Acción '{action}' no implementada."}}

@app.get("/api/accounts/status")
@verify_token
async def get_accounts_status(request: Request):
    user_id = request.state.user["uid"]
    if not db: raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        accounts_ref = db.collection("users").document(user_id).collection("connected_accounts")
        accounts = [doc.id for doc in accounts_ref.stream()]
        return {"connected": accounts}
    except Exception as e:
        return {"connected": []}

# --- 5. ENDPOINTS DE OAUTH2 ---
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback"

@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token")
    if not id_token: raise HTTPException(status_code=400, detail="Falta el token de usuario.")
    scope = "https://www.googleapis.com/auth/gmail.readonly"
    url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={REDIRECT_URI_GOOGLE}"
           f"&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={id_token}")
    return RedirectResponse(url=url)

@app.get("/google/callback")
async def google_callback(request: Request):
    initialize_firebase_admin_once()
    if not db: raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    id_token = request.query_params.get("state"); code = request.query_params.get("code")
    if not id_token or not code: raise HTTPException(status_code=400, detail="Falta información en el callback.")

    try:
        user_id = auth.verify_id_token(id_token)["uid"]
        token_url = "https://oauth2.googleapis.com/token"
        data = {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "code": code,
                "redirect_uri": REDIRECT_URI_GOOGLE, "grant_type": "authorization_code"}
        r = requests.post(token_url, data=data); r.raise_for_status(); tokens = r.json()
        
        user_ref = db.collection("users").document(user_id)
        user_ref.collection("connected_accounts").document("google").set(tokens)
        
        return JSONResponse(content={"status": "Cuenta de Google conectada. Puedes cerrar esta ventana."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")