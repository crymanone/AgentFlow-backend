import os
import json
from functools import wraps
import base64
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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

# Variable global para la base de datos Firestore
db = None

# Inicializa Firebase Admin SDK solo una vez para evitar errores en entornos serverless.
def initialize_firebase_admin_once():
    global db
    if not firebase_admin._apps:
        try:
            # Lee las credenciales desde una variable de entorno (más seguro)
            cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"]
            cred_dict = json.loads(cred_json_str)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(cred)
            db = firestore.client()
            print("Firebase Admin SDK inicializado con éxito.")
        except Exception as e:
            # Este es un error crítico que debe ser registrado
            print(f"ERROR CRÍTICO: No se pudo inicializar Firebase Admin SDK: {e}")
            db = None # Aseguramos que db sea None si falla la inicialización

# Configuración de los clientes de IA
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("ADVERTENCIA: GEMINI_API_KEY no encontrada. Las funciones de Gemini fallarán.")

try:
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
except KeyError:
    print("ADVERTENCIA: OPENAI_API_KEY no encontrada. Las funciones de OpenAI fallarán.")
    openai_client = None

# --- Creación de la Aplicación FastAPI ---
app = FastAPI(
    title="AgentFlow Production Backend",
    description="Backend para gestionar la lógica de IA y la conexión con servicios de terceros.",
    version="9.0.0" # Versión final estable
)

# Configuración de CORS para permitir peticiones desde el frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # En producción, sería mejor restringirlo al dominio del frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# 2. AUTENTICACIÓN Y GESTIÓN DE CREDENCIALES
# ==============================================================================

# Decorador para verificar el token de Firebase en las cabeceras de las peticiones
def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        initialize_firebase_admin_once() # Aseguramos que Firebase esté listo
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Falta cabecera de autenticación.")
        
        id_token = auth_header.split("Bearer ")[1]
        try:
            # Verificamos el token con Firebase Auth
            decoded_token = auth.verify_id_token(id_token)
            request.state.user = decoded_token # Adjuntamos la info del usuario a la petición
        except auth.InvalidIdTokenError:
            raise HTTPException(status_code=403, detail="El token de sesión es inválido.")
        except Exception as e:
            raise HTTPException(status_code=403, detail=f"Error de autenticación: {e}")
        return await f(request, *args, **kwargs)
    return decorated

# [SOLUCIÓN CLAVE] Función mejorada para obtener y refrescar tokens de Google
def get_google_service(user_id: str, service_name: str, version: str, scopes: list):
    if not db: raise Exception("La base de datos no está disponible.")
    
    doc_ref = db.collection("users").document(user_id).collection("connected_accounts").document("google")
    doc = doc_ref.get()
    if not doc.exists:
        raise Exception("El usuario no ha conectado su cuenta de Google.")
    
    tokens = doc.to_dict()
    creds = Credentials.from_authorized_user_info(tokens, scopes)

    # Lógica de refresco del token
    if not creds.valid:
        if creds.expired and creds.refresh_token:
            print(f"Token de Google para {user_id} expirado. Refrescando...")
            creds.refresh(GoogleAuthRequest())
            # [LA CORRECCIÓN MÁS IMPORTANTE]
            # Guardamos el objeto completo de credenciales actualizado,
            # no solo el access_token. Esto preserva el nuevo id_token,
            # la fecha de expiración y cualquier otro dato relevante.
            new_token_data = {
                'token': creds.token,
                'refresh_token': creds.refresh_token,
                'token_uri': creds.token_uri,
                'client_id': creds.client_id,
                'client_secret': creds.client_secret,
                'scopes': creds.scopes
            }
            doc_ref.set(new_token_data) # Usamos 'set' para sobreescribir con la info fresca
            print(f"Token para {user_id} refrescado y guardado con éxito.")
        else:
            raise Exception("No se pueden refrescar las credenciales de Google. El usuario necesita reconectarse.")
    
    return build(service_name, version, credentials=creds)

# ==============================================================================
# 3. LÓGICA DE INTELIGENCIA ARTIFICIAL (IA)
# ==============================================================================

# [IA MEJORADA] Utiliza GPT-4o para una interpretación de intenciones más precisa.
def interpret_intent_with_openai(text: str) -> dict:
    if not openai_client:
        raise Exception("El cliente de OpenAI no está inicializado. Verifica la API Key.")
    
    system_prompt = """
    Eres "AuraOS", un sistema operativo de inteligencia artificial ultrapreciso. Tu función es analizar el comando de un usuario y traducirlo a un objeto JSON estructurado. No respondas con texto, solo con el JSON.

    Acciones válidas:
    - "summarize_inbox": Para resúmenes generales de correos.
    - "search_emails": Para buscar correos específicos.
    - "create_draft": Para redactar un nuevo correo.
    - "find_contact": Para buscar la información de un contacto.
    - "create_event": Para agendar una cita en el calendario.
    - "unknown": Si la intención no es clara o no se puede realizar.

    Parámetros que puedes extraer:
    - "query_text": El texto principal de una búsqueda de correos (ej: "facturas de Vercel").
    - "time_period": Periodos relativos (ej: "hoy", "ayer", "la última semana", "el mes pasado"). Normalízalos a "today", "yesterday", "last_week", "last_month".
    - "sender_name": El nombre del remitente (ej: "Carlos").
    - "recipient_email": El email del destinatario para un nuevo correo.
    - "recipient_name": El nombre del destinatario (para buscar su email).
    - "content_summary": El objetivo o resumen del cuerpo de un correo a redactar.
    - "contact_name": El nombre del contacto a buscar.
    - "event_summary": El título o resumen del evento a crear.
    - "event_date_time": La descripción de la fecha y hora del evento (ej: "mañana a las 10:30am").
    - "event_location": La ubicación física o virtual del evento.
    - "attendees": Una lista de nombres o correos de los asistentes.

    Ejemplos:
    - "resume los correos de ayer de sofia" -> {"action": "search_emails", "parameters": {"time_period": "yesterday", "sender_name": "sofia"}}
    - "crea un borrador para elon@tesla.com sobre la fusión" -> {"action": "create_draft", "parameters": {"recipient_email": "elon@tesla.com", "content_summary": "informar sobre la fusión"}}
    - "agendame una reunion con jeff mañana a las 3pm en la sala principal" -> {"action": "create_event", "parameters": {"event_summary": "Reunión con Jeff", "attendees": ["jeff"], "event_date_time": "mañana a las 3pm", "event_location": "sala principal"}}

    Si un parámetro no está presente, omítelo del JSON. Responde únicamente con el objeto JSON.
    """
    try:
        completion = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        response_content = completion.choices[0].message.content
        return json.loads(response_content) if response_content else {"action": "unknown"}
    except Exception as e:
        print(f"Error en OpenAI: {e}")
        return {"action": "error", "parameters": {"message": f"IA (OpenAI) no pudo interpretar la intención: {e}"}}

# [IA MEJORADA] Gemini ahora usa el contexto completo para redactar correos de alta calidad.
def generate_draft_with_gemini(params: dict, original_command: str) -> dict:
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
    Eres Aura, una asistente ejecutiva de IA experta en comunicación profesional.
    Tu tarea es redactar un borrador de correo electrónico basado en la petición de un usuario.

    COMANDO ORIGINAL DEL USUARIO: "{original_command}"
    
    OBJETIVO DEL CORREO: "{params.get("content_summary", "No especificado")}"

    Instrucciones:
    1.  Infiere el tono (formal, informal, urgente) a partir del comando original.
    2.  Crea un asunto (subject) claro, conciso y profesional.
    3.  Redacta el cuerpo (body) del correo. Sé completo pero ve al grano.
    4.  Si el comando es muy breve, expande el contenido de forma lógica y profesional.
    5.  Responde ÚNICA Y EXCLUSIVAMENTE con un objeto JSON válido que contenga las claves "subject" y "body". No incluyas texto adicional ni explicaciones.
    
    Ejemplo de respuesta JSON:
    {{
      "subject": "Seguimiento sobre nuestra reunión",
      "body": "Hola [Nombre],\\n\\nEspero que estés bien.\\n\\nQuería hacer un seguimiento de nuestra conversación de ayer sobre el proyecto X.\\n\\nQuedo a tu disposición.\\n\\nSaludos,"
    }}
    """
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(response_mime_type="application/json")
        )
        return json.loads(response.text)
    except Exception as e:
        print(f"Error en Gemini (Draft Generation): {e}")
        # Plan B: si Gemini falla, crea un borrador básico para no dejar al usuario sin nada.
        return {
            "subject": f"Borrador: {params.get('content_summary', 'Asunto pendiente')}",
            "body": f"Hola,\n\nEste es un borrador generado a partir de la petición: '{original_command}'.\n\nPor favor, complétalo.\n\nSaludos,"
        }


# [IA MEJORADA] Gemini ahora es mucho más preciso para interpretar fechas complejas.
def parse_datetime_with_gemini(text_date: str) -> str:
    model = genai.GenerativeModel('gemini-2.5-pro')
    prompt = f"""
    Tu única tarea es analizar un texto que describe una fecha y hora y convertirlo a un formato ISO 8601 UTC estricto (YYYY-MM-DDTHH:MM:SSZ).
    Asume que la fecha de hoy es {datetime.now().strftime('%Y-%m-%d')}.
    Interpreta frases relativas como "mañana a las 10:30", "lunes que viene a las 3pm", "el 29".

    EJEMPLOS DE CONVERSIÓN:
    - "mañana a las 10:30 de la mañana" -> (calcula la fecha de mañana)T10:30:00Z
    - "el día 22 de octubre a las 10am" -> (año actual)-10-22T10:00:00Z
    - "lunes 29 por la mañana en la sala 1" -> (calcula la fecha del próximo lunes 29)T09:00:00Z (asume una hora de oficina si no se especifica)
    - "dentro de 3 dias a las 4 de la tarde" -> (calcula la fecha)T16:00:00Z

    Si el texto es ambiguo o no es una fecha válida (ej: "pronto"), devuelve un string que contenga la palabra "ERROR".
    RESPONDE ÚNICA Y EXCLUSIVAMENTE CON EL STRING ISO 8601 O UN STRING CON "ERROR".

    Texto a analizar: '{text_date}'
    """
    try:
        response = model.generate_content(prompt)
        iso_date = response.text.strip().replace("`", "")
        if "ERROR" in iso_date: return ""
        # Verificación final para asegurar que parece una fecha ISO válida
        datetime.fromisoformat(iso_date.replace("Z", "+00:00"))
        return iso_date
    except Exception as e:
        print(f"Error parseando fecha '{text_date}' con Gemini: {e}")
        return "" # Devolvemos vacío si la IA devuelve algo incorrecto o hay un error


# ==============================================================================
# 4. LÓGICA DE INTEGRACIÓN CON GOOGLE (Gmail, Calendar, Contacts)
# ==============================================================================

def search_google_contacts(user_id: str, contact_name: str) -> dict:
    """Busca un contacto y devuelve su email principal."""
    try:
        service = get_google_service(user_id, 'people', 'v1', scopes=["https://www.googleapis.com/auth/contacts.readonly"])
        results = service.people().searchContacts(
            query=contact_name,
            readMask="emailAddresses,names"
        ).execute()
        
        contacts = results.get('results', [])
        if not contacts:
            raise Exception(f"No he encontrado a nadie llamado '{contact_name}' en tus contactos.")
        
        # Por ahora, devolvemos el primer resultado que tenga email.
        for person_result in contacts:
            person = person_result.get('person', {})
            emails = person.get('emailAddresses', [])
            if emails:
                return {"name": person.get('names', [{}])[0].get('displayName', 'N/A'), "email": emails[0].get('value')}
        
        raise Exception(f"Encontré a '{contact_name}', pero no tiene una dirección de correo guardada.")

    except HttpError as e:
        raise Exception(f"Error con la API de Contactos de Google: {e.reason}")

def create_draft_in_gmail(user_id: str, draft_data: dict):
    """Crea un borrador en la cuenta de Gmail del usuario."""
    try:
        if not draft_data.get('to'):
             raise ValueError("El destinatario (to) es obligatorio para crear un borrador.")

        service = get_google_service(user_id, 'gmail', 'v1', scopes=["https://www.googleapis.com/auth/gmail.compose"])
        
        message = MIMEText(draft_data.get('body', ''))
        message['to'] = draft_data['to']
        message['subject'] = draft_data.get('subject', '(Sin asunto)')
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("utf-8")
        
        draft_body = {'message': {'raw': raw_message}}
        created_draft = service.users().drafts().create(userId='me', body=draft_body).execute()
        return created_draft
    except HttpError as e:
        error_details = json.loads(e.content.decode('utf-8'))
        raise Exception(f"Error de la API de Gmail: {error_details['error']['message']}")
    except Exception as e:
        raise Exception(f"Error creando el borrador: {e}")
        
def send_draft_from_gmail(user_id: str, draft_id: str):
    """Envía un borrador existente de Gmail."""
    try:
        service = get_google_service(user_id, 'gmail', 'v1', scopes=['https://www.googleapis.com/auth/gmail.compose'])
        service.users().drafts().send(userId='me', body={'id': draft_id}).execute()
        return {"status": "success", "message": "Correo enviado con éxito."}
    except HttpError as error:
        error_details = json.loads(error.content.decode('utf-8'))
        raise Exception(f"Error de la API de Gmail al enviar: {error_details['error']['message']}")


def create_event_in_calendar(user_id: str, event_details: dict):
    """Crea un evento en el calendario principal del usuario."""
    try:
        service = get_google_service(user_id, 'calendar', 'v3', scopes=['https://www.googleapis.com/auth/calendar.events'])
        
        date_text = event_details.get("event_date_time", "")
        if not date_text:
             raise ValueError("La fecha y hora son necesarias para crear un evento.")

        start_time_str = parse_datetime_with_gemini(date_text)
        if not start_time_str:
            raise ValueError(f"No pude entender la fecha y hora '{date_text}'. Por favor, sé más específico (ej: 'mañana a las 10:30' o '22 de octubre a las 10am').")

        start_time_dt = datetime.fromisoformat(start_time_str.replace("Z", "+00:00"))
        end_time_dt = start_time_dt + timedelta(hours=1) # Duración por defecto: 1 hora

        attendees_list = []
        if event_details.get("attendees"):
            for attendee_name in event_details["attendees"]:
                try:
                    contact_info = search_google_contacts(user_id, attendee_name)
                    attendees_list.append({'email': contact_info['email']})
                except Exception as e:
                    print(f"No se pudo encontrar el email del asistente '{attendee_name}': {e}")
        
        event = {
            'summary': event_details.get('event_summary', 'Evento de AgentFlow'),
            'location': event_details.get('event_location', ''),
            'description': f'Evento creado por Aura a partir del comando: "{event_details.get("original_command")}"',
            'start': {'dateTime': start_time_dt.isoformat(), 'timeZone': 'UTC'},
            'end': {'dateTime': end_time_dt.isoformat(), 'timeZone': 'UTC'},
            'attendees': attendees_list,
        }

        created_event = service.events().insert(calendarId='primary', body=event, sendNotifications=True).execute()
        return {"message": f"Evento '{created_event.get('summary')}' creado.", "url": created_event.get('htmlLink')}
    except HttpError as e:
        raise Exception(f"Error con la API de Google Calendar: {e.reason}")
    except Exception as e:
        raise e # Re-lanzamos la excepción para que el endpoint principal la capture

# ==============================================================================
# 5. ENDPOINTS DE LA API (Lógica Principal)
# ==============================================================================

@app.get("/")
def root():
    return {"status": "AgentFlow Backend Activo y Operacional"}

class CommandPayload(BaseModel):
    text: str

@app.post("/api/voice-command")
@verify_token
async def voice_command(request: Request, data: CommandPayload):
    user_id = request.state.user["uid"]
    text = data.text
    
    try:
        intent = interpret_intent_with_openai(text)
        action = intent.get("action", "unknown")
        params = intent.get("parameters", {})
        params['original_command'] = text

        if action == "create_draft":
            # Si el destinatario es un nombre, buscar su email primero
            if params.get("recipient_name") and not params.get("recipient_email"):
                contact_info = search_google_contacts(user_id, params["recipient_name"])
                params["recipient_email"] = contact_info['email']
            
            # Generar el contenido del correo con IA
            email_content = generate_draft_with_gemini(params, text)
            full_draft_data = {
                "to": params.get("recipient_email"),
                "subject": email_content.get("subject"),
                "body": email_content.get("body")
            }
            # Crear el borrador en Gmail
            created_draft = create_draft_in_gmail(user_id, full_draft_data)
            full_draft_data['id'] = created_draft.get('id')
            return {"action": "draft_created", "payload": {"draft": full_draft_data}}

        elif action == "create_event":
            result = create_event_in_calendar(user_id, params)
            return {"action": "event_created", "payload": result}

        # ... (Aquí irían las demás acciones como "search_emails", "summarize_inbox", etc.) ...

        else: # Acción desconocida o de error
            return {"action": "unknown", "payload": {"message": f"No he entendido la acción '{action}'. Inténtalo de nuevo."}}

    except Exception as e:
        # Captura cualquier excepción y la devuelve como un error estructurado
        print(f"ERROR en /api/voice-command para el usuario {user_id}: {e}")
        return JSONResponse(
            status_code=400,
            content={"action": "error", "payload": {"message": str(e)}}
        )


@app.post("/api/drafts/send/{draft_id}")
@verify_token
async def send_draft(request: Request, draft_id: str):
    user_id = request.state.user["uid"]
    try:
        result = send_draft_from_gmail(user_id, draft_id)
        return {"action": "draft_sent", "payload": result}
    except Exception as e:
        print(f"ERROR al enviar borrador {draft_id} para el usuario {user_id}: {e}")
        return JSONResponse(
            status_code=400,
            content={"action": "error", "payload": {"message": str(e)}}
        )

# ==============================================================================
# 6. ENDPOINTS DE CONEXIÓN DE CUENTAS (OAuth2)
# ==============================================================================

class AuthCode(BaseModel):
    code: str

@app.post("/api/connect/google")
@verify_token
async def connect_google_account(request: Request, data: AuthCode):
    """
    Recibe el 'code' de autorización del frontend, lo intercambia por tokens
    de acceso y refresco, y los guarda en Firestore.
    """
    user_id = request.state.user["uid"]
    code = data.code
    if not db:
        raise HTTPException(status_code=500, detail="La base de datos no está inicializada.")

    try:
        token_url = "https://oauth2.googleapis.com/token"
        payload = {
            "client_id": os.environ.get("GOOGLE_CLIENT_ID"),
            "client_secret": os.environ.get("GOOGLE_CLIENT_SECRET"),
            "code": code,
            "redirect_uri": "https://agent-flow-backend-drab.vercel.app/google/callback", # Esta debe coincidir EXACTAMENTE con la de la consola de Google
            "grant_type": "authorization_code"
        }
        
        r = requests.post(token_url, data=payload)
        r.raise_for_status() # Lanza una excepción si la respuesta no es 2xx
        tokens = r.json()
        
        # Guarda los tokens de forma segura en Firestore
        user_ref = db.collection("users").document(user_id)
        user_ref.collection("connected_accounts").document("google").set(tokens)

        return {"status": "success", "message": "Cuenta de Google conectada con éxito."}

    except requests.exceptions.HTTPError as e:
        print(f"ERROR HTTP al intercambiar código: {e.response.text}")
        raise HTTPException(status_code=400, detail=f"No se pudo verificar con Google. Es posible que el código haya expirado. Por favor, intenta conectar de nuevo.")
    except Exception as e:
        print(f"ERROR finalizando conexión: {e}")
        raise HTTPException(status_code=500, detail=f"Ocurrió un error inesperado al vincular la cuenta: {e}")


@app.get("/api/accounts/status")
@verify_token
async def get_accounts_status(request: Request):
    user_id = request.state.user["uid"]
    if not db:
        return {"connected": []} # Si no hay DB, no hay cuentas conectadas
    try:
        accounts_ref = db.collection("users").document(user_id).collection("connected_accounts")
        accounts = [doc.id for doc in accounts_ref.stream()]
        return {"connected": accounts}
    except Exception as e:
        print(f"Error al obtener estado de cuentas para {user_id}: {e}")
        return {"connected": []}

@app.get("/google/callback")
async def google_callback(request: Request):
    """
    Endpoint de redirección para el flujo de OAuth2. Cierra la ventana del navegador.
    La lógica principal se maneja en /api/connect/google.
    """
    return JSONResponse(content={
        "status": "completed",
        "message": "Proceso de autorización completado. Puedes cerrar esta ventana y volver a la aplicación."
    })