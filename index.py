import os
import json
from functools import wraps
import base64
from datetime import datetime, timezone
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

app = FastAPI(title="AgentFlow Production Backend v5.1")
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
    model = genai.GenerativeModel('gemini-1.5-flash-latest')
    prompt = f"""
    Analiza: "{text}". Extrae 'action' y 'parameters' en JSON.
    ACCIONES: "summarize_inbox", "search_emails", "create_draft", "find_contact", "check_availability", "create_event", "unknown".
    PARÁMETROS: "client_name", "time_period", "recipient", "content_summary", "contact_name", "event_summary", "event_date_time".
    EJEMPLOS:
    - "busca correos de Acme" -> {{"action": "search_emails", "parameters": {{"client_name": "Acme"}}}}
    - "escribe a jefe@empresa.com que el informe está listo" -> {{"action": "create_draft", "parameters": {{"recipient": "jefe@empresa.com", "content_summary": "informar que el informe está listo"}}}}
    - "cuál es el email de Ana García" -> {{"action": "find_contact", "parameters": {{"contact_name": "Ana García"}}}}
    - "tengo hueco mañana por la tarde" -> {{"action": "check_availability", "parameters": {{"time_period": "tomorrow afternoon"}}}}
    - "crea un evento con Juan Pérez el viernes a las 10 para revisar el proyecto" -> {{"action": "create_event", "parameters": {{"attendee_name": "Juan Pérez", "event_date_time": "next Friday at 10am", "event_summary": "revisar el proyecto"}}}}
    Responde SÓLO con el JSON.
    """
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e: return {"action": "error", "parameters": {"message": f"IA no pudo interpretar: {e}"}}

def generate_draft_with_gemini(params: dict) -> dict:
    content_summary = params.get("content_summary", "No se especificó contenido.")
    prompt = f"""
    Actúa como Aura. Escribe un correo profesional basado en el siguiente objetivo: "{content_summary}".
    Responde en JSON con "subject" y "body".
    """
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return json.loads(response.text)

def get_google_service(user_id: str, service_name: str, version: str, scopes: list):
    if not db: raise Exception("Base de datos no disponible.")
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get()
    if not doc.exists: raise Exception("Cuenta de Google no conectada.")
    
    tokens = doc.to_dict()
    creds = Credentials(token=tokens.get("access_token"), refresh_token=tokens.get("refresh_token"),
                        token_uri="https://oauth2.googleapis.com/token", client_id=os.environ.get("GOOGLE_CLIENT_ID"),
                        client_secret=os.environ.get("GOOGLE_CLIENT_SECRET"), scopes=scopes)

    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        doc_ref.update({"access_token": creds.token})
    
    return build(service_name, version, credentials=creds)

def translate_params_to_gmail_query(params: dict) -> str:
    query_parts = []
    if params.get("client_name"): query_parts.append(f'from:"{params["client_name"]}"')
    if params.get("time_period"):
        period_map = {"hoy": "newer_than:1d", "ayer": "older_than:1d newer_than:2d", "last week": "newer_than:7d", "ultimo mes": "newer_than:30d"}
        query_parts.append(period_map.get(params["time_period"], ""))
    query_parts.extend(["-in:trash", "-in:spam"])
    return " ".join(filter(None, query_parts))
    
def get_real_emails_for_user(user_id: str, search_query: str = "") -> list:
    try:
        service = get_google_service(user_id, 'gmail', 'v1', scopes=["https://www.googleapis.com/auth/gmail.readonly"])
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
    except HttpError as error: raise Exception(f"Error de API de Gmail: {error.reason}")

def summarize_emails_with_gemini(emails: list) -> str:
    prompt = f'Eres Aura. Resume estos correos de forma ejecutiva: {json.dumps(emails)}'
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text.strip()
    
def create_draft_in_gmail(user_id: str, draft_data: dict):
    try:
        service = get_google_service(user_id, 'gmail', 'v1', scopes=["https://www.googleapis.com/auth/gmail.compose"])
        message_body = (f"To: {draft_data['to']}\n"
                        f"Subject: {draft_data['subject']}\n\n"
                        f"{draft_data['body']}")
        raw_message = base64.urlsafe_b64encode(message_body.encode("utf-8")).decode("utf-8")
        draft = {'message': {'raw': raw_message}}
        created_draft = service.users().drafts().create(userId='me', body=draft).execute()
        return created_draft
    except HttpError as error:
        raise Exception(f"Error de API de Gmail al crear borrador: {error.reason}")

def find_contact_in_google(user_id: str, contact_name: str):
    try:
        service = get_google_service(user_id, 'people', 'v1', scopes=["https://www.googleapis.com/auth/contacts.readonly"])
        results = service.people().searchContacts(query=contact_name, readMask="emailAddresses,names").execute()
        contacts = results.get('results', [])
        if not contacts:
            return {"type": "not_found", "data": {"message": f"No se encontró a nadie llamado '{contact_name}'."}}
        if len(contacts) > 1:
            options = [{"name": p.get('person', {}).get('names', [{}])[0].get('displayName', 'N/A'),
                        "email": p.get('person', {}).get('emailAddresses', [{}])[0].get('value', 'N/A')}
                       for p in contacts]
            return {"type": "disambiguation", "data": {"question": "¿A cuál te refieres?", "options": options}}
        person = contacts[0].get('person', {}); name = person.get('names', [{}])[0].get('displayName', 'N/A'); email = person.get('emailAddresses', [{}])[0].get('value', 'N/A')
        return {"type": "found", "data": {"name": name, "email": email}}
    except HttpError as error: raise Exception(f"Error de API de Contactos: {error.reason}")

def get_calendar_events(user_id: str):
    try:
        service = get_google_service(user_id, 'calendar', 'v3', scopes=["https://www.googleapis.com/auth/calendar.readonly"])
        now = datetime.utcnow().isoformat() + 'Z'
        events_result = service.events().list(calendarId='primary', timeMin=now, maxResults=5, singleEvents=True, orderBy='startTime').execute()
        events = events_result.get('items', [])
        if not events: return {"message": "No tienes próximos eventos."}
        event_list = [f"- {event['summary']} (a las {datetime.fromisoformat(event['start'].get('dateTime')).strftime('%H:%M')})" for event in events]
        return {"events": "\n".join(event_list)}
    except HttpError as error: raise Exception(f"Error de API de Calendario: {error.reason}")

# --- 4. ENDPOINTS DE LA API ---
@app.get("/")
def root(): return {"status": "AgentFlow Backend Activo"}

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]; text = data.get("text", "")
    intent = interpret_intent_with_gemini(text); action = intent.get("action"); params = intent.get("parameters", {})
    try:
        if action == "summarize_inbox" or action == "search_emails":
            query = translate_params_to_gmail_query(params)
            emails = get_real_emails_for_user(user_id, search_query=query)
            if action == "summarize_inbox":
                if not emails: return {"action": "summarize_inbox", "payload": {"summary": "No hay correos que coincidan."}}
                summary = summarize_emails_with_gemini(emails)
                return {"action": "summarize_inbox", "payload": {"summary": summary}}
            else:
                return {"action": "search_emails_result", "payload": {"emails": emails}}
        elif action == "create_draft":
            email_content = generate_draft_with_gemini(params)
            full_draft_data = {"to": params.get("recipient"), "subject": email_content.get("subject"), "body": email_content.get("body")}
            created_draft = create_draft_in_gmail(user_id, full_draft_data)
            full_draft_data['id'] = created_draft.get('id')
            return {"action": "draft_created", "payload": {"draft": full_draft_data}}
        elif action == "find_contact":
            contact_result = find_contact_in_google(user_id, params.get("contact_name"))
            if contact_result["type"] == "disambiguation":
                return {"action": "ask", "payload": contact_result["data"]}
            else:
                return {"action": "contact_found", "payload": contact_result["data"]}
        elif action == "check_availability":
            events = get_calendar_events(user_id)
            return {"action": "availability_checked", "payload": events}
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
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback"

@app.get("/auth/google")
async def auth_google(request: Request):
    id_token = request.query_params.get("token")
    scopes = [
        "https://www.googleapis.com/auth/gmail.compose",
        "https://www.googleapis.com/auth/calendar.events.readonly",
        "https://www.googleapis.com/auth/contacts.readonly"
    ]
    scope_string = " ".join(scopes)
    url = (f"https://accounts.google.com/o/oauth2/v2/auth?client_id={GOOGLE_CLIENT_ID}&redirect_uri={REDIRECT_URI_GOOGLE}"
           f"&response_type=code&scope={scope_string}&access_type=offline&prompt=consent&state={id_token}")
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")