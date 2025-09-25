import os
import json
from functools import wraps
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import base64

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
            # Confiamos en que Vercel inyecta las variables
            cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"] 
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
        except Exception as e:
            print(f"ERROR CRÍTICO inicializando Firebase Admin: {e}")

try: genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError: print("ADVERTENCIA: API Key de Gemini no encontrada.")

app = FastAPI(title="AgentFlow Production Backend v4.5")
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
    Analiza: "{text}". Extrae 'action' ("summarize_inbox", "search_emails", "create_draft") y 'parameters' ("client_name", "time_period", "recipient", "content_summary") en JSON.
    Ejemplo 1: "resume correos de acme" -> {{"action": "summarize_inbox", "parameters": {{"client_name": "acme"}}}}
    Ejemplo 2: "escribe a jefe@empresa.com que el informe está listo" -> {{"action": "create_draft", "parameters": {{"recipient": "jefe@empresa.com", "content_summary": "informar que el informe está listo"}}}}
    Responde SÓLO con el JSON.
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e: return {"action": "error", "parameters": {"message": f"IA no pudo interpretar: {e}"}}

def generate_draft_body_with_gemini(params: dict) -> dict:
    content_summary = params.get("content_summary", "No se especificó contenido.")
    prompt = f"""
    Actúa como Aura, una asistente de IA profesional. Tu tarea es escribir un correo electrónico.
    OBJETIVO DEL CORREO: "{content_summary}"
    Escribe un correo que sea claro, conciso y profesional. No incluyas el destinatario (To:).
    Formato de respuesta: JSON estricto con las claves "subject" y "body".
    Ejemplo: {{"subject": "Título del Correo", "body": "Cuerpo del mensaje..."}}
    """
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    response = model.generate_content(prompt)
    return json.loads(response.text)

def get_real_emails_for_user(user_id: str, search_query: str = "") -> list:
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
    
    # ... Lógica de get emails (sin cambios)
    return [] # Placeholder

def summarize_emails_with_gemini(emails: list) -> str:
    # ... (Sin cambios)
    return "" # Placeholder
    
def create_draft_in_gmail(user_id: str, draft_data: dict):
    if not db: raise Exception("Base de datos no disponible.")
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get();
    if not doc.exists: raise Exception("Cuenta de Google no conectada.")
    
    tokens = doc.to_dict()
    creds = Credentials(token=tokens.get("access_token"), refresh_token=tokens.get("refresh_token"),
                        token_uri="https://oauth2.googleapis.com/token", client_id=os.environ.get("GOOGLE_CLIENT_ID"),
                        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"), scopes=["https://www.googleapis.com/auth/gmail.compose"])

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest()); doc_ref.update({"access_token": creds.token})

    try:
        service = build('gmail', 'v1', credentials=creds)
        message_body = (f"To: {draft_data['to']}\n"
                        f"Subject: {draft_data['subject']}\n\n"
                        f"{draft_data['body']}")
        raw_message = base64.urlsafe_b64encode(message_body.encode("utf-8")).decode("utf-8")
        draft = {'message': {'raw': raw_message}}
        created_draft = service.users().drafts().create(userId='me', body=draft).execute()
        return created_draft
    except HttpError as error:
        raise Exception(f"Error de API de Gmail al crear borrador: {error.reason}")

# --- ENDPOINTS ---
@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]; text = data.get("text", "")
    intent = interpret_intent_with_gemini(text); action = intent.get("action"); params = intent.get("parameters", {})
    try:
        if action == "create_draft":
            email_content = generate_draft_body_with_gemini(params)
            full_draft_data = {"to": params.get("recipient"), "subject": email_content.get("subject"), "body": email_content.get("body")}
            created_draft = create_draft_in_gmail(user_id, full_draft_data)
            full_draft_data['id'] = created_draft.get('id')
            return {"action": "draft_created", "payload": {"draft": full_draft_data}}
        
        # (Lógica para otras acciones)
        else:
             return {"action": "unknown", "payload": {"message": f"Acción '{action}' no implementada."}}

    except Exception as e: return {"action": "error", "payload": {"message": str(e)}}

# ... (Endpoints de OAuth y otros sin cambios)
@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token")
    
    # [LA CORRECCIÓN] Leemos directamente, si falla aquí, lo veremos en los logs
    client_id = os.environ["GOOGLE_CLIENT_ID"]
    redirect_uri = os.environ["REDIRECT_URI_GOOGLE"] # Asumimos que esta también está

    print(f"--- OAUTH CON CLIENT ID: {client_id[:5]}... ---") # Imprimimos los 5 primeros caracteres

    scope = "https://www.googleapis.com/auth/gmail.readonly"
    url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={client_id}&redirect_uri={redirect_uri}"
           f"&response_type=code&scope={scope}&access_type=offline&prompt=consent&state={id_token}")
    return RedirectResponse(url=url)

@app.get("/api/accounts/status")
@verify_token
async def get_accounts_status(request: Request):
    user_id = request.state.user["uid"]
    if not db:
        raise HTTPException(status_code=500, detail="Base de datos no disponible.")
    try:
        accounts_ref = db.collection("users").document(user_id).collection("connected_accounts")
        accounts = [doc.id for doc in accounts_ref.stream()]
        print(f"Cuentas encontradas para {user_id}: {accounts}")
        return {"connected": accounts}
    except Exception as e:
        print(f"Error al obtener estado de cuentas para {user_id}: {e}")
        return {"connected": []}    