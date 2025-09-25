import os
import json
from functools import wraps
import base64
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

app = FastAPI(title="AgentFlow Production Backend v4.7")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- 2. DECORADOR DE AUTENTICACIÓN ---
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

# --- 3. LÓGICA DE IA Y DATOS ---
def interpret_intent_with_gemini(text: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Analiza: "{text}". Extrae 'action' ("summarize_inbox", "search_emails", "create_draft") y 'parameters' ("client_name", "time_period", "recipient", "content_summary") en JSON.
    Ejemplo 1: "resume correos de acme" -> {{"action": "summarize_inbox", "parameters": {{"client_name": "acme"}}}}
    Ejemplo 2: "escribe a jefe@empresa.com que el informe está listo" -> {{"action": "create_draft", "parameters": {{"recipient": "jefe@empresa.com", "content_summary": "informar que el informe está listo"}}}}
    Responde SÓLO con el JSON.
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e: return {"action": "error", "parameters": {"message": f"IA no pudo interpretar: {e}"}}

def generate_draft_with_gemini(params: dict) -> dict:
    content_summary = params.get("content_summary", "No se especificó contenido.")
    prompt = f"""
    Actúa como Aura, una asistente de IA profesional. Tu tarea es escribir un correo electrónico.
    OBJETIVO DEL CORREO: "{content_summary}"
    Escribe un correo que sea claro, conciso y profesional. No incluyas el destinatario (To:).
    Formato de respuesta: JSON estricto con las claves "subject" y "body".
    Ejemplo: {{"subject": "Actualización del Informe", "body": "Hola,\\n\\nSolo para confirmarte que el informe está casi listo. Lo tendrás mañana a primera hora.\\n\\nSaludos,\\nAura"}}
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return json.loads(response.text)
    
def get_gmail_service(user_id: str, write_permission: bool = False):
    if not db: raise Exception("Base de datos no disponible.")
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get();
    if not doc.exists: raise Exception("Cuenta de Google no conectada.")
    
    tokens = doc.to_dict()
    scopes = ["https://www.googleapis.com/auth/gmail.compose"] if write_permission else ["https://www.googleapis.com/auth/gmail.readonly"]
    creds = Credentials(token=tokens.get("access_token"), refresh_token=tokens.get("refresh_token"),
                        token_uri="https://oauth2.googleapis.com/token", client_id=os.environ.get("GOOGLE_CLIENT_ID"),
                        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"), scopes=scopes)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest()); doc_ref.update({"access_token": creds.token})
    
    return build('gmail', 'v1', credentials=creds)

def create_draft_in_gmail(user_id: str, draft_data: dict):
    try:
        service = get_gmail_service(user_id, write_permission=True)
        message_body = (f"To: {draft_data['to']}\n"
                        f"Subject: {draft_data['subject']}\n\n"
                        f"{draft_data['body']}")
        raw_message = base64.urlsafe_b64encode(message_body.encode("utf-8")).decode("utf-8")
        draft = {'message': {'raw': raw_message}}
        created_draft = service.users().drafts().create(userId='me', body=draft).execute()
        return created_draft
    except HttpError as error:
        raise Exception(f"Error de API de Gmail al crear borrador: {error.reason}")

def send_draft_from_gmail(user_id: str, draft_id: str):
    try:
        service = get_gmail_service(user_id, write_permission=True)
        sent_message = service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
        return sent_message
    except HttpError as error:
        raise Exception(f"Error de API de Gmail al enviar borrador: {error.reason}")

# --- 4. ENDPOINTS DE LA API ---
@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]; text = data.get("text", "")
    intent = interpret_intent_with_gemini(text); action = intent.get("action"); params = intent.get("parameters", {})
    try:
        if action == "create_draft":
            email_content = generate_draft_with_gemini(params)
            full_draft_data = {"to": params.get("recipient"), "subject": email_content.get("subject"), "body": email_content.get("body")}
            created_draft = create_draft_in_gmail(user_id, full_draft_data)
            full_draft_data['id'] = created_draft.get('id')
            return {"action": "draft_created", "payload": {"draft": full_draft_data}}
        # (Aquí iría la lógica de 'summarize' y 'search' que hemos omitido por brevedad)
        else:
            return {"action": "unknown", "payload": {"message": f"Acción '{action}' no implementada."}}
    except Exception as e:
        return {"action": "error", "payload": {"message": str(e)}}

@app.post("/api/drafts/send/{draft_id}")
@verify_token
async def send_draft(request: Request, draft_id: str):
    user_id = request.state.user["uid"]
    try:
        sent_message = send_draft_from_gmail(user_id, draft_id)
        return {"status": "success", "message_id": sent_message.get('id')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- (El resto de endpoints no han cambiado y se omiten por brevedad) ---
@app.get("/")
def root(): return {"status": "AgentFlow Backend Activo"}

@app.get("/api/accounts/status")
@verify_token
async def get_accounts_status(request: Request):
    user_id = request.state.user["uid"]; #...
    return {"connected": []} # Placeholder

GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback"

@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token")
    scope = "https://www.googleapis.com/auth/gmail.compose"
    url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={REDIRECT_URI_GOOGLE}"
           f"&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={id_token}")
    return RedirectResponse(url=url)

@app.get("/google/callback")
async def google_callback(request: Request):
    initialize_firebase_admin_once()
    id_token = request.query_params.get("state"); code = request.query_params.get("code")
    try:
        user_id = auth.verify_id_token(id_token)["uid"]
        token_url = "https://oauth2.googleapis.com/token"
        data = {"client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET, "code": code,
                "redirect_uri": REDIRECT_URI_GOOGLE, "grant_type": "authorization_code"}
        r = requests.post(token_url, data=data); r.raise_for_status(); tokens = r.json()
        
        user_ref = db.collection("users").document(user_id)
        user_ref.collection("connected_accounts").document("google").set(tokens)
        return JSONResponse(content={"status": "Cuenta de Google conectada."})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")