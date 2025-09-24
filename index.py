import os
import json
from functools import wraps
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests

# --- SDKs ---
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request as GoogleAuthRequest

# --- INICIALIZACIÓN ---
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

try: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError: print("ADVERTENCIA: API Key de Gemini no encontrada.")

app = FastAPI(title="AgentFlow Production Backend v4.3")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- DECORADOR DE AUTENTICACIÓN ---
def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        initialize_firebase_admin_once()
        id_token = request.headers.get("Authorization", "").split("Bearer ")[-1]
        try:
            request.state.user = auth.verify_id_token(id_token)
        except Exception as e: raise HTTPException(status_code=403, detail=f"Token inválido: {e}")
        return await f(request, *args, **kwargs)
    return decorated

# --- LÓGICA DE IA Y DATOS ---
def interpret_intent_with_gemini(text: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Analiza el siguiente comando y extráelo a un formato JSON estricto.
    Comando: "{text}"
    
    ACCIONES VÁLIDAS: "summarize_inbox", "search_emails", "unknown".
    PARÁMETROS VÁLIDOS:
    - "client_name": string
    - "time_period": string, normalizado a uno de ["today", "yesterday", "last_7_days", "last_30_days"].

    REGLAS DE NORMALIZACIÓN DE FECHAS:
    - "hoy", "de hoy" -> "today"
    - "ayer", "de ayer" -> "yesterday"
    - "esta semana", "última semana", "últimos 7 días" -> "last_7_days"
    - "este mes", "último mes", "últimos 30 días" -> "last_30_days"

    EJEMPLOS:
    - "resume los correos de hoy" -> {{"action": "summarize_inbox", "parameters": {{"time_period": "today"}}}}
    - "busca emails de 'ACME Corp' del ultimo mes" -> {{"action": "search_emails", "parameters": {{"client_name": "ACME Corp", "time_period": "last_30_days"}}}}
    
    RESPONDE ÚNICA Y EXCLUSIVAMENTE CON EL OBJETO JSON.
    """
    try:
        response = model.generate_content(prompt)
        print(f"Respuesta cruda de Gemini (Intención): {response.text}")
        return json.loads(response.text)
    except Exception as e:
        print(f"Error en Gemini interpretando intención: {e}")
        return {"action": "error", "parameters": {"message": "IA no pudo interpretar."}}

def translate_params_to_gmail_query(params: dict) -> str:
    query_parts = []
    if params.get("client_name"): query_parts.append(f'from:"{params["client_name"]}"')
    if params.get("time_period"):
        period_map = {"today": "newer_than:1d", "yesterday": "older_than:1d newer_than:2d", "last_7_days": "newer_than:7d", "last_30_days": "newer_than:30d"}
        query_parts.append(period_map.get(params["time_period"], ""))
    query_parts.extend(["in:inbox", "-in:trash", "-in:spam"])
    return " ".join(filter(None, query_parts))

def get_real_emails_for_user(user_id: str, search_query: str = None) -> list:
    if not db: raise Exception("Base de datos no disponible.")
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get()
    if not doc.exists: raise Exception("Cuenta de Google no conectada.")
    
    tokens = doc.to_dict()
    creds = Credentials(token=tokens.get("access_token"), refresh_token=tokens.get("refresh_token"),
                        token_uri="https://oauth2.googleapis.com/token", client_id=os.environ.get("GOOGLE_CLIENT_ID"),
                        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"), scopes=["https://www.googleapis.com/auth/gmail.readonly"])

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest()); doc_ref.update({"access_token": creds.token})

    try:
        service = build('gmail', 'v1', credentials=creds)
        if search_query:
            results = service.users().messages().list(userId='me', q=search_query, maxResults=10).execute()
        else:
            results = service.users().messages().list(userId='me', labelIds=['INBOX'], maxResults=5).execute()
        
        messages = results.get('messages', [])
        emails_list = []
        if messages:
            for message in messages:
                msg = service.users().messages().get(userId='me', id=message['id'], format='metadata', metadataHeaders=['From', 'Subject']).execute()
                headers = msg.get('payload', {}).get('headers', [])
                subject = next((i['value'] for i in headers if i['name'] == 'Subject'), 'Sin Asunto')
                sender = next((i['value'] for i in headers if i['name'] == 'From'), 'Desconocido')
                emails_list.append({"from": sender, "subject": subject, "snippet": msg.get('snippet', '')})
        return emails_list
    except HttpError as error: raise Exception(f"Error de API de Gmail: {error.reason}")

def summarize_emails_with_gemini(emails: list) -> str:
    prompt = f'Eres Aura. Resume estos correos de forma ejecutiva: {json.dumps(emails)}'
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return response.text.strip()

# --- ENDPOINTS ---
@app.get("/")
def root(): return {"status": "AgentFlow Backend Activo"}

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]; text = data.get("text", "")
    intent = interpret_intent_with_gemini(text); action = intent.get("action"); params = intent.get("parameters", {})
    try:
        query = translate_params_to_gmail_query(params)
        if action == "summarize_inbox":
            emails = get_real_emails_for_user(user_id, search_query=query)
            if not emails: return {"action": "summarize_inbox", "payload": {"summary": "No hay correos que coincidan con tu petición."}}
            summary = summarize_emails_with_gemini(emails)
            return {"action": "summarize_inbox", "payload": {"summary": summary}}
        
        elif action == "search_emails":
            emails = get_real_emails_for_user(user_id, search_query=query)
            return {"action": "search_emails_result", "payload": {"emails": emails}}
            
        elif action == "error": return {"action": "error", "payload": params}
        else: return {"action": "unknown", "payload": {"message": f"Acción '{action}' no implementada."}}
    except Exception as e: return {"action": "error", "payload": {"message": str(e)}}

@app.get("/api/accounts/status")
@verify_token
async def get_accounts_status(request: Request):
    user_id = request.state.user["uid"]
    if not db: raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        accounts_ref = db.collection("users").document(user_id).collection("connected_accounts")
        accounts = [doc.id for doc in accounts_ref.stream()]
        return {"connected": accounts}
    except Exception: return {"connected": []}

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback"

@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token")
    scope = "https://www.googleapis.com/auth/gmail.readonly"
    url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={REDIRECT_URI_GOOGLE}"
           f"&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={id_token}")
    return RedirectResponse(url=url)

@app.get("/google/callback")
async def google_callback(request: Request):
    initialize_firebase_admin_once()
    if not db: raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    id_token = request.query_params.get("state"); code = request.query_params.get("code")
    try:
        user_id = auth.verify_id_token(id_token)["uid"]
        token_url = "https://oauth2.googleapis.com/token"
        data = {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "code": code,
                "redirect_uri": REDIRECT_URI_GOOGLE, "grant_type": "authorization_code"}
        r = requests.post(token_url, data=data); r.raise_for_status(); tokens = r.json()
        
        user_ref = db.collection("users").document(user_id)
        user_ref.collection("connected_accounts").document("google").set(tokens)
        return JSONResponse(content={"status": "Cuenta de Google conectada. Puedes cerrar esta ventana."})
    except Exception as e: raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")