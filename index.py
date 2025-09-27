# ==============================================================================
# AgentFlow Backend - Versión 14.0 (Autenticación Estable + IA Mejorada)
# CEO: Cryman09
# CTO: Gemini
# ==============================================================================

import os
import json
from functools import wraps
import base64
from datetime import datetime, date, timezone, timedelta
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse, RedirectResponse, HTMLResponse
import requests
from email.mime.text import MIMEText

# --- SDKs y Librerías ---
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore
from openai import OpenAI
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth.transport.requests import Request as GoogleAuthRequest

# ==============================================================================
# 1. INICIALIZACIÓN DE SERVICIOS GLOBALES
# ==============================================================================

db = None
openai_client = None

def initialize_firebase_admin_once():
    global db
    if not firebase_admin._apps:
        try:
            cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"]
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase Admin SDK inicializado con éxito.")
        except Exception as e:
            print(f"ERROR CRÍTICO: No se pudo inicializar Firebase Admin SDK: {e}")
            db = None

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("ADVERTENCIA: GEMINI_API_KEY no encontrada.")

try:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    print("Cliente de OpenAI inicializado correctamente.")
except KeyError:
    print("ADVERTENCIA CRÍTICA: OPENAI_API_KEY no encontrada.")
    openai_client = None

app = FastAPI(title="AgentFlow Production Backend", version="14.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ==============================================================================
# 2. AUTENTICACIÓN Y GESTIÓN DE CREDENCIALES
# ==============================================================================

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

def get_google_service(user_id: str, service_name: str, version: str, scopes: list):
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get()
    if not doc.exists: raise Exception("Cuenta de Google no conectada.")
    
    config = doc.to_dict()
    # Corrige el campo 'token' a 'access_token' si es necesario para compatibilidad
    if 'token' in config and 'access_token' not in config:
        config['access_token'] = config.pop('token')

    creds = Credentials.from_authorized_user_info(config, scopes)
    if not creds.valid and creds.expired and creds.refresh_token:
        creds.refresh(GoogleAuthRequest())
        # Guarda usando el formato que espera Credentials.from_authorized_user_info
        tokens_to_save = {
            'access_token': creds.token, 'refresh_token': creds.refresh_token, 
            'token_uri': creds.token_uri, 'client_id': creds.client_id, 
            'client_secret': creds.client_secret, 'scopes': creds.scopes,
            'expiry': creds.expiry.isoformat()
        }
        doc_ref.set(tokens_to_save)
    return build(service_name, version, credentials=creds)

# ==============================================================================
# 3. LÓGICA DE INTELIGENCIA ARTIFICIAL (IA) - [SECCIÓN MEJORADA]
# ==============================================================================

def interpret_intent_with_openai(text: str) -> dict:
    if not openai_client: raise Exception("El servicio de interpretación de IA no está disponible.")
    system_prompt = """
    Eres "AuraOS", un sistema de IA. Analiza el comando y tradúcelo a un JSON.
    Acciones: "summarize_inbox", "search_emails", "create_draft", "create_event", "unknown".
    Parámetros: "query_text", "time_period", "sender_name", "recipient_email", "recipient_name", "content_summary", "event_summary", "event_date_time", "attendees", "content_date_reference".
    "content_date_reference" es CUALQUIER mención de fecha/día en un comando para crear un correo.
    Ej: "escribe a jefe@empresa.com que el lunes no voy" -> {"action": "create_draft", "parameters": { "recipient_email": "jefe@empresa.com", "content_summary": "informar que no iré a trabajar", "content_date_reference": "el lunes" }}
    Responde solo con el JSON.
    """
    try:
        completion = openai_client.chat.completions.create(model="gpt-4o", messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": text}], response_format={"type": "json_object"})
        return json.loads(completion.choices[0].message.content or "{}")
    except Exception as e: raise Exception(f"La IA no pudo procesar la petición: {e}")

def generate_draft_with_gemini(params: dict, original_command: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    concrete_date = params.get("concrete_date", "")
    date_instruction = f"Si el comando original menciona una fecha, insértala de forma natural en el texto. La fecha concreta es: {concrete_date}." if concrete_date else "No insertes placeholders como '[insertar fecha]'."
    prompt = f"""
    Eres Aura. Redacta un borrador de correo.
    COMANDO: "{original_command}"
    OBJETIVO: "{params.get("content_summary", "No especificado")}"
    FECHA: {date_instruction}
    Instrucciones: Crea "subject" y "body" profesionales. Responde solo con un JSON.
    """
    try:
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)
    except Exception as e: return {"subject": f"Borrador: {params.get('content_summary', '')}", "body": f"Petición: '{original_command}' (error de IA: {e})"}

def parse_datetime_for_calendar(text_date: str) -> dict:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Analiza un texto de fecha/hora. Hoy es {datetime.now().strftime('%Y-%m-%d')}.
    Responde solo con JSON con "iso_date" (YYYY-MM-DDTHH:MM:SSZ) y "time_specified" (true/false).
    Ej: "mañana a las 10am" -> {{"iso_date": "(mañana)T10:00:00Z", "time_specified": true}}
    Ej: "el día 30" -> {{"iso_date": "(día 30)T09:00:00Z", "time_specified": false}}
    Si es ambiguo, devuelve {{}}. Texto: '{text_date}'
    """
    try:
        response = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
        return json.loads(response.text)
    except Exception: return {}

def parse_date_for_email(text_date: str) -> str:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f"""
    Convierte un texto de fecha a formato legible (ej: 'lunes, 29 de septiembre de 2025').
    Hoy es {datetime.now().strftime('%Y-%m-%d')}. Si es ambiguo, devuelve "".
    Responde solo con el string de la fecha. Texto: '{text_date}'
    """
    try:
        response = model.generate_content(prompt); date_str = response.text.strip().replace("`", "")
        return "" if "ERROR" in date_str else date_str
    except Exception: return ""

def summarize_emails_with_gemini(emails: list) -> str:
    model = genai.GenerativeModel('gemini-1.5-pro-latest')
    prompt = f'Eres Aura. Resume estos correos de forma ejecutiva y concisa:\n\n{json.dumps(emails)}'
    return model.generate_content(prompt).text.strip()

# ==============================================================================
# 4. LÓGICA DE INTEGRACIÓN CON GOOGLE
# ==============================================================================

def translate_params_to_gmail_query(params: dict) -> str:
    query_parts = []
    if params.get("query_text"): query_parts.append(params["query_text"])
    if params.get("sender_name"): query_parts.append(f'from:"{params["sender_name"]}"')
    if params.get("time_period"):
        period_map = {"today": "newer_than:1d", "yesterday": "older_than:1d newer_than:2d", "last_week": "newer_than:7d"}
        query_parts.append(period_map.get(params["time_period"], ""))
    return " ".join(filter(None, query_parts))

def get_real_emails_for_user(user_id: str, search_query: str = "", max_results: int = 10) -> list:
    service = get_google_service(user_id, 'gmail', 'v1', scopes=["https://www.googleapis.com/auth/gmail.readonly"])
    results = service.users().messages().list(userId='me', q=search_query, maxResults=max_results).execute()
    messages = results.get('messages', [])
    emails_list = []
    for msg_info in messages:
        msg = service.users().messages().get(userId='me', id=msg_info['id'], format='metadata', metadataHeaders=['From', 'Subject']).execute()
        headers = msg.get('payload', {}).get('headers', [])
        subject = next((h['value'] for h in headers if h['name'] == 'Subject'), '(Sin Asunto)')
        sender = next((h['value'] for h in headers if h['name'] == 'From'), 'Desconocido')
        emails_list.append({"id": msg['id'], "from": sender, "subject": subject, "snippet": msg.get('snippet', '')})
    return emails_list

def search_google_contacts(user_id: str, contact_name: str) -> dict:
    service = get_google_service(user_id, 'people', 'v1', scopes=["https://www.googleapis.com/auth/contacts.readonly"])
    results = service.people().searchContacts(query=contact_name, readMask="emailAddresses,names").execute()
    contacts = results.get('results', [])
    if not contacts: raise Exception(f"No encontré a '{contact_name}'.")
    for person_result in contacts:
        person = person_result.get('person', {})
        if person.get('emailAddresses'): return {"name": person.get('names', [{}])[0].get('displayName', 'N/A'), "email": person['emailAddresses'][0].get('value')}
    raise Exception(f"Encontré a '{contact_name}', pero sin correo.")

def create_draft_in_gmail(user_id: str, draft_data: dict):
    service = get_google_service(user_id, 'gmail', 'v1', scopes=["https://www.googleapis.com/auth/gmail.compose"])
    message = MIMEText(draft_data.get('body', '')); message['to'] = draft_data['to']; message['subject'] = draft_data.get('subject', '(Sin asunto)')
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
    return service.users().drafts().create(userId='me', body={'message': {'raw': raw_message}}).execute()

def send_draft_from_gmail(user_id: str, draft_id: str):
    service = get_google_service(user_id, 'gmail', 'v1', scopes=['https://www.googleapis.com/auth/gmail.compose'])
    return service.users().drafts().send(userId='me', body={'id': draft_id}).execute()

def create_event_in_calendar(user_id: str, event_details: dict):
    service = get_google_service(user_id, 'calendar', 'v3', scopes=['https://www.googleapis.com/auth/calendar.events'])
    date_text = event_details.get("event_date_time", "")
    if not date_text: raise ValueError("Falta la fecha para el evento.")
    parsed_info = parse_datetime_for_calendar(date_text)
    if not parsed_info.get("iso_date"): raise ValueError(f"No entendí la fecha '{date_text}'.")
    
    event = {'summary': event_details.get('event_summary', 'Evento de AgentFlow'), 'location': event_details.get('event_location', '')}
    if parsed_info.get("time_specified"):
        start_dt = datetime.fromisoformat(parsed_info["iso_date"].replace("Z", "+00:00"))
        event.update({'start': {'dateTime': start_dt.isoformat(), 'timeZone': 'UTC'}, 'end': {'dateTime': (start_dt + timedelta(hours=1)).isoformat(), 'timeZone': 'UTC'}})
    else:
        start_date = datetime.fromisoformat(parsed_info["iso_date"].replace("Z", "+00:00")).date()
        event.update({'start': {'date': start_date.isoformat()}, 'end': {'date': (start_date + timedelta(days=1)).isoformat()}})
    
    created_event = service.events().insert(calendarId='primary', body=event).execute()
    return {"message": f"Evento '{created_event.get('summary')}' creado.", "url": created_event.get('htmlLink')}

# ==============================================================================
# 5. ENDPOINTS DE LA API (Lógica Principal)
# ==============================================================================

@app.get("/")
def root(): return {"status": "AgentFlow Backend Activo y Operacional"}

class CommandPayload(BaseModel): text: str

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: CommandPayload):
    user_id = request.state.user["uid"]; text = data.text
    try:
        intent = interpret_intent_with_openai(text)
        action = intent.get("action", "unknown"); params = intent.get("parameters", {})
        if action == "unknown": raise Exception("No he entendido tu petición.")

        if action == "search_emails":
            query = translate_params_to_gmail_query(params)
            emails = get_real_emails_for_user(user_id, search_query=query)
            return {"action": "search_emails_result", "payload": {"emails": emails}}
        
        elif action == "summarize_inbox":
            query = translate_params_to_gmail_query(params)
            emails = get_real_emails_for_user(user_id, search_query=query, max_results=5)
            if not emails: return {"action": "summarize_inbox", "payload": {"summary": "No hay correos que coincidan."}}
            summary = summarize_emails_with_gemini(emails)
            return {"action": "summarize_inbox", "payload": {"summary": summary}}

        elif action == "create_draft":
            if params.get("content_date_reference"):
                params["concrete_date"] = parse_date_for_email(params["content_date_reference"])
            if params.get("recipient_name") and not params.get("recipient_email"):
                params["recipient_email"] = search_google_contacts(user_id, params["recipient_name"])['email']
            email_content = generate_draft_with_gemini(params, text)
            full_draft_data = {"to": params.get("recipient_email"), **email_content}
            created_draft = create_draft_in_gmail(user_id, full_draft_data)
            full_draft_data['id'] = created_draft.get('id')
            return {"action": "draft_created", "payload": {"draft": full_draft_data}}
        
        elif action == "create_event":
            result = create_event_in_calendar(user_id, params)
            return {"action": "event_created", "payload": result}
        
        else: raise Exception(f"La acción '{action}' no está implementada.")
    except Exception as e:
        return JSONResponse(status_code=400, content={"action": "error", "payload": {"message": str(e)}})

@app.post("/api/drafts/send/{draft_id}")
@verify_token
async def send_draft(request: Request, draft_id: str):
    user_id = request.state.user["uid"]
    try:
        send_draft_from_gmail(user_id, draft_id)
        return {"action": "draft_sent", "payload": {"message": "Correo enviado."}}
    except Exception as e:
        return JSONResponse(status_code=400, content={"action": "error", "payload": {"message": str(e)}})


# ==============================================================================
# 6. ENDPOINTS DE CONEXIÓN DE CUENTAS (OAuth2)
# ==============================================================================

class AuthPayload(BaseModel):
    code: str
    redirectUri: str

@app.post("/api/connect/google")
@verify_token
async def connect_google_account(request: Request, data: AuthPayload):
    user_id = request.state.user["uid"]
    code = data.code
    frontend_redirect_uri = data.redirectUri
    
    try:
        # VALIDACIÓN ADICIONAL: Verifica que la redirect_uri sea válida
        allowed_redirect_uris = [
            "http://localhost:8081",
            "exp://localhost:8081",
            "yourapp://google-callback"  # Agrega tu esquema de deep linking
        ]
        
        if frontend_redirect_uri not in allowed_redirect_uris:
            print(f"Redirect URI no permitida: {frontend_redirect_uri}")
            raise HTTPException(status_code=400, detail="Redirect URI no válida")
        
        payload = {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
            "code": code,
            "redirect_uri": frontend_redirect_uri,
            "grant_type": "authorization_code"
        }
        
        print(f"Intercambiando código por token para usuario {user_id}")
        
        r = requests.post("https://oauth2.googleapis.com/token", data=payload)
        r.raise_for_status()
        tokens = r.json()
        
        # Guarda solo los tokens esenciales
        tokens_to_save = {
            'token': tokens.get('access_token'),
            'refresh_token': tokens.get('refresh_token'),
            'token_uri': 'https://oauth2.googleapis.com/token',
            'client_id': os.environ.get("GOOGLE_CLIENT_ID"),
            'client_secret': os.environ.get("GOOGLE_CLIENT_SECRET"),
            'scopes': tokens.get('scope', '').split(' '),
            'expiry': (datetime.now(timezone.utc) + timedelta(seconds=tokens.get('expires_in', 3600))).isoformat()
        }
        
        db.collection("users").document(user_id).collection("connected_accounts").document("google").set(tokens_to_save)
        
        print(f"Cuenta de Google conectada exitosamente para usuario {user_id}")
        return {"status": "success", "message": "Cuenta conectada exitosamente"}
        
    except requests.exceptions.HTTPError as e:
        error_detail = f"Error de Google OAuth: {e.response.status_code} - {e.response.text}"
        print(error_detail)
        raise HTTPException(status_code=400, detail=error_detail)
    except Exception as e:
        error_detail = f"Error inesperado: {str(e)}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=error_detail)

@app.get("/api/accounts/status")
@verify_token
async def get_accounts_status(request: Request):
    """Endpoint para verificar el estado de las cuentas conectadas - VERSIÓN CORREGIDA"""
    user_id = request.state.user["uid"]
    
    print(f"🔍 Verificando estado de cuentas para usuario: {user_id}")
    
    # ✅ Asegurar que Firebase esté inicializado
    initialize_firebase_admin_once()
    
    if not db:
        print("❌ Base de datos no disponible")
        return {"connected": []}
    
    try:
        accounts_ref = db.collection("users").document(user_id).collection("connected_accounts")
        docs = accounts_ref.stream()
        
        connected_accounts = []
        for doc in docs:
            print(f"✅ Cuenta encontrada: {doc.id}")
            connected_accounts.append(doc.id)
        
        print(f"📊 Cuentas conectadas para {user_id}: {connected_accounts}")
        return {"connected": connected_accounts}
        
    except Exception as e:
        print(f"❌ Error verificando estado de cuentas: {e}")
        return {"connected": []}


@app.get("/google/auth-start")
async def google_auth_start(request: Request):
    """Endpoint que inicia el flujo OAuth de Google"""
    try:
        # ✅ INICIALIZAR FIREBASE PRIMERO
        initialize_firebase_admin_once()
        
        firebase_token = request.query_params.get("firebaseToken")
        
        if not firebase_token:
            return JSONResponse(
                status_code=400, 
                content={"error": "Token de Firebase requerido"}
            )
        
        # Verificar el token de Firebase
        decoded_token = auth.verify_id_token(firebase_token)
        user_id = decoded_token["uid"]
        
        print(f"✅ Iniciando OAuth para usuario: {user_id}")
        
        # Construir la URL de autorización de Google
        scopes = [
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.compose", 
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/contacts.readonly"
        ]
        
        google_auth_url = (
            "https://accounts.google.com/o/oauth2/v2/auth?"
            f"client_id={os.environ.get('GOOGLE_CLIENT_ID')}&"
            f"redirect_uri=https://agent-flow-backend-drab.vercel.app/google/callback&"
            f"response_type=code&"
            f"scope={'%20'.join(scopes)}&"
            f"access_type=offline&"
            f"prompt=consent&"
            f"state={user_id}"
        )
        
        print(f"🔗 Redirigiendo a: {google_auth_url}")
        return RedirectResponse(google_auth_url)
        
    except Exception as e:
        print(f"❌ Error en /google/auth-start: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Error iniciando autenticación: {str(e)}"}
        )
        

@app.get("/google/callback")
async def google_callback(request: Request):
    """Callback de Google OAuth"""
    try:
        # ✅ INICIALIZAR FIREBASE PRIMERO
        initialize_firebase_admin_once()
        
        code = request.query_params.get("code")
        state = request.query_params.get("state")  # user_id
        
        print(f"📥 Callback recibido - code: {code}, state: {state}")
        
        if not code:
            return HTMLResponse(content="""
            <html>
                <body style="background: #0D142E; color: white; text-align: center; padding: 50px; font-family: Arial;">
                    <h1>❌ Error</h1>
                    <p>No se recibió el código de autorización.</p>
                    <button onclick="window.close()" style="background: #FF6B6B; padding: 10px 20px; border: none; border-radius: 5px; color: white; cursor: pointer;">
                        Cerrar
                    </button>
                </body>
            </html>
            """)

        # Intercambiar código por tokens
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": "https://agent-flow-backend-drab.vercel.app/google/callback"
        }
        
        print("🔄 Intercambiando código por tokens...")
        response = requests.post(token_url, data=payload)
        response.raise_for_status()
        tokens = response.json()
        
        print("✅ Tokens recibidos de Google")
        
        # Guardar tokens en Firestore
        if state and db:
            tokens_to_save = {
                'token': tokens.get('access_token'),
                'refresh_token': tokens.get('refresh_token'),
                'client_id': os.environ.get("GOOGLE_CLIENT_ID"),
                'client_secret': os.environ.get("GOOGLE_CLIENT_SECRET"),
                'scopes': tokens.get('scope', '').split(),
                'expiry': (datetime.now(timezone.utc) + timedelta(seconds=tokens.get('expires_in', 3600))).isoformat(),
                'connected_at': datetime.now(timezone.utc).isoformat()
            }
            
            db.collection("users").document(state).collection("connected_accounts").document("google").set(tokens_to_save)
            print(f"✅ Cuenta de Google conectada para usuario {state}")

        # Página de éxito
        success_html = """
        <html>
            <head>
                <title>✅ Conexión Exitosa</title>
                <script>
                    function closeWindow() {
                        if (window.history.length > 1) {
                            window.history.back();
                        } else {
                            window.close();
                        }
                    }
                    setTimeout(() => {
                        closeWindow();
                    }, 2000);
                </script>
            </head>
            <body style="background: #0D142E; color: white; text-align: center; padding: 50px; font-family: Arial;">
                <h1 style="color: #4CAF50;">✅ ¡Conexión Exitosa!</h1>
                <p>Tu cuenta de Google ha sido conectada correctamente.</p>
                <p>Esta ventana se cerrará automáticamente en 2 segundos...</p>
                <div style="margin-top: 30px;">
                    <button onclick="closeWindow()" style="background: #1E90FF; color: white; border: none; padding: 12px 24px; border-radius: 8px; cursor: pointer; font-size: 16px;">
                        Cerrar Ventana Ahora
                    </button>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=success_html)
        
    except Exception as e:
        print(f"❌ Error en callback: {str(e)}")
        error_html = f"""
        <html>
            <body style="background: #0D142E; color: white; text-align: center; padding: 50px; font-family: Arial;">
                <h1 style="color: #FF6B6B;">❌ Error de Conexión</h1>
                <p>Ha ocurrido un error: {str(e)}</p>
                <p>Por favor, intenta nuevamente.</p>
                <button onclick="window.close()" style="background: #FF6B6B; color: white; border: none; padding: 10px 20px; border-radius: 5px; cursor: pointer;">
                    Cerrar y Reintentar
                </button>
            </body>
        </html>
        """
        return HTMLResponse(content=error_html)  

@app.get("/api/debug/google-status")
@verify_token
async def debug_google_status(request: Request):
    """Endpoint de diagnóstico para verificar el estado de Google"""
    user_id = request.state.user["uid"]
    
    print(f"🔍 Diagnóstico Google para usuario: {user_id}")
    
    initialize_firebase_admin_once()
    
    if not db:
        return {"error": "Database not available"}
    
    try:
        # Verificar si existe el documento de Google
        google_doc = db.collection("users").document(user_id).collection("connected_accounts").document("google").get()
        
        if google_doc.exists:
            data = google_doc.to_dict()
            return {
                "connected": True,
                "has_token": bool(data.get('token')),
                "has_refresh_token": bool(data.get('refresh_token')),
                "scopes": data.get('scopes', []),
                "connected_at": data.get('connected_at')
            }
        else:
            return {"connected": False, "reason": "Google document does not exist"}
            
    except Exception as e:
        return {"connected": False, "error": str(e)}          