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

# --- 1. INICIALIZACIÓN Y CONFIGURACIÓN ---
db = None

def get_env_variable(var_name: str) -> str:
    """Función segura para obtener variables de entorno."""
    try:
        return os.environ[var_name]
    except KeyError:
        raise Exception(f"Variable de entorno requerida no encontrada: '{var_name}'")

def initialize_firebase_admin_once():
    global db
    if not firebase_admin._apps:
        try:
            cred_json_str = get_env_variable("FIREBASE_ADMIN_SDK_JSON")
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase Admin SDK inicializado.")
        except Exception as e:
            print(f"ERROR CRÍTICO inicializando Firebase Admin: {e}")

try:
    genai.configure(api_key=get_env_variable("GEMINI_API_KEY"))
except Exception as e:
    print(f"ADVERTENCIA: {e}")

app = FastAPI(title="AgentFlow Production Backend v3.9")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 2. DECORADOR DE AUTENTICACIÓN ---
def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        initialize_firebase_admin_once()
        id_token = request.headers.get("Authorization", "").split("Bearer ")[-1]
        try:
            request.state.user = auth.verify_id_token(id_token)
        except Exception as e:
            raise HTTPException(status_code=403, detail=f"Token inválido: {e}")
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
    Proporciona ÚNICA Y EXCLUSIVAMENTE el objeto JSON correspondiente. No añadas texto extra ni explicaciones.
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
        period_map = {"hoy": "newer_than:1d", "ayer": "older_than:1d newer_than:2d", "last week": "newer_than:7d", "ultimo mes": "newer_than:30d"}
        query_parts.append(period_map.get(params["time_period"], ""))
    query_parts.extend(["-in:trash", "-in:spam"])
    return " ".join(filter(None, query_parts))

def get_real_emails_for_user(user_id: str, search_query: str = "") -> list:
    if not db: raise Exception("Base de datos no disponible.")
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get()
    if not doc.exists: raise Exception("Cuenta de Google no conectada.")
    
    tokens = doc.to_dict()
    client_id = get_env_variable("GOOGLE_CLIENT_ID")
    client_secret = get_env_variable("GOOGLE_CLIENT_SECRET")
    creds = Credentials(token=tokens.get("access_token"), refresh_token=tokens.get("refresh_token"),
                        token_uri="https://oauth2.googleapis.com/token", client_id=client_id,
                        client_secret=client_secret, scopes=["https://www.googleapis.com/auth/gmail.readonly"])

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        doc_ref.update({"access_token": creds.token})
    try:
        service = build('gmail', 'v1', credentials=creds)
        results = service.users().messages().list(userId='me', q=search_query, maxResults=10).execute()
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
    except HttpError as error:
        raise Exception(f"Error de API de Gmail: {error.reason}")

def summarize_emails_with_gemini(emails: list) -> str:
    prompt = f'Eres Aura. Resume estos correos de forma ejecutiva: {json.dumps(emails)}'
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
    intent = interpret_intent_with_gemini(text); action = intent.get("action"); params = intent.get("parameters", {})
    try:
        query = translate_params_to_gmail_query(params)
        if action == "summarize_inbox" or action == "search_emails":
            emails = get_real_emails_for_user(user_id, search_query=query)
            if action == "summarize_inbox":
                if not emails: return {"action": "summarize_inbox", "payload": {"summary": "No hay correos que coincidan."}}
                summary = summarize_emails_with_gemini(emails)
                return {"action": "summarize_inbox", "payload": {"summary": summary}}
            else:
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

# --- 5. ENDPOINTS DE OAUTH2 ---
@app.get("/auth/google")
async def auth_google(request: Request):
    try:
        print("--- Iniciando /auth/google ---")
        id_token = request.query_params.get("token")
        if not id_token:
            raise HTTPException(status_code=400, detail="Falta el token de usuario.")

        client_id = get_env_variable("GOOGLE_CLIENT_ID")
        redirect_uri = get_env_variable("REDIRECT_URI_GOOGLE")

        print(f"Client ID y Redirect URI cargados correctamente.")
        scope = "https://www.googleapis.com/auth/gmail.readonly"
        url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&redirect_uri={redirect_uri}"
               f"&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={id_token}")
        
        return RedirectResponse(url=url)
    except Exception as e:
        print(f"!!!!!!!! ERROR EN /auth/google !!!!!!!!!!")
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/google/callback")
async def google_callback(request: Request):
    try:
        print("--- Iniciando /google/callback ---")
        initialize_firebase_admin_once()
        if not db: raise Exception("La base de datos no está disponible.")
        
        id_token = request.query_params.get("state")
        code = request.query_params.get("code")
        if not id_token or not code:
            raise HTTPException(status_code=400, detail="Falta información en el callback.")

        user_id = auth.verify_id_token(id_token)["uid"]
        print(f"Callback para UID: {user_id}")

        client_id = get_env_variable("GOOGLE_CLIENT_ID")
        client_secret = get_env_variable("GOOGLE_CLIENT_SECRET")
        redirect_uri = get_env_variable("REDIRECT_URI_GOOGLE")
        
        token_url = "https://oauth2.googleapis.com/token"
        data = {"client_id": client_id, "client_secret": client_secret, "code": code,
                "redirect_uri": redirect_uri, "grant_type": "authorization_code"}
        
        r = requests.post(token_url, data=data)
        r.raise_for_status()
        tokens = r.json()
        print("Tokens de Google obtenidos correctamente.")
        
        user_ref = db.collection("users").document(user_id)
        user_ref.collection("connected_accounts").document("google").set(tokens)
        print(f"Tokens de Google guardados en Firestore para UID: {user_id}")
        
        return JSONResponse(content={"status": "Cuenta de Google conectada. Puedes cerrar esta ventana."})
    except Exception as e:
        print(f"!!!!!!!! ERROR EN /google/callback !!!!!!!!!!")
        print(str(e))
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")