import os
import json
from functools import wraps
from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import RedirectResponse, JSONResponse
import requests

# --- SDKs de Google y Firebase ---
import google.generativeai as genai
import firebase_admin
from firebase_admin import credentials, auth, firestore

# --- 1. INICIALIZACIÓN DE SERVICIOS ---

# Inicializa Firebase Admin SDK (para verificar tokens y acceder a Firestore)
try:
    cred_json_str = os.environ["FIREBASE_ADMIN_SDK_JSON"]
    cred_dict = json.loads(cred_json_str)
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)
    db = firestore.client()
    print("Firebase Admin SDK inicializado correctamente.")
except KeyError:
    db = None
    print("ADVERTENCIA: Credenciales de Firebase Admin no encontradas. La autenticación y Firestore no funcionarán.")

# Inicializa Gemini
try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    print("API de Gemini configurada.")
except KeyError:
    print("ADVERTENCIA: API Key de Gemini no encontrada. La IA no funcionará.")

app = FastAPI(title="AgentFlow Production Backend")


# --- 2. DECORADOR DE AUTENTICACIÓN (NUESTRA MURALLA DE SEGURIDAD) ---

def verify_token(f):
    @wraps(f)
    async def decorated(request: Request, *args, **kwargs):
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Falta el token de autorización o el formato es incorrecto.")
        
        id_token = auth_header.split("Bearer ")[1]
        
        try:
            decoded_token = auth.verify_id_token(id_token)
            request.state.user = decoded_token # Inyectamos los datos del usuario en la petición
        except Exception as e:
            raise HTTPException(status_code=403, detail=f"Token inválido o expirado: {e}")
        
        return await f(request, *args, **kwargs)
    return decorated

# --- 3. ENDPOINTS ---

# No necesitamos un endpoint /login, Firebase lo gestiona desde el cliente.
# La app obtiene el token y lo envía a nuestros endpoints protegidos.

@app.get("/")
def root():
    return {"status": "AgentFlow Backend Activo"}


@app.post("/api/voice-command")
@verify_token # <--- Endpoint protegido
async def voice_command(request: Request, data: dict):
    user_id = request.state.user["uid"]
    text = data.get("text", "")
    
    # --- Llamada al cerebro de Gemini para interpretar la intención ---
    model = genai.GenerativeModel('gemini-1.5-pro-latest', generation_config=genai.types.GenerationConfig(response_mime_type="application/json"))
    prompt = f"""
    Analiza el siguiente comando y extráe la 'action' y los 'parameters'.
    Comando: "{text}"
    Acciones posibles: "summarize_inbox", "search_emails", "create_draft".
    Parámetros a extraer: "client_name", "time_period".
    Responde ÚNICAMENTE con un JSON válido.
    Ejemplo: si el comando es "resume correos de acme de ayer", responde {{"action": "summarize_inbox", "parameters": {{"client_name": "acme", "time_period": "yesterday"}}}}
    """
    try:
        response = model.generate_content(prompt)
        intent = json.loads(response.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al procesar el comando con la IA: {e}")
    
    action = intent.get("action")
    parameters = intent.get("parameters", {})

    # --- ACTION ENGINE (Motor de Acciones) ---
    if action == "summarize_inbox":
        # Placeholder: Aquí iría la lógica para leer correos de Gmail
        # 1. Leer de Firestore las credenciales de Gmail para user_id
        # 2. Conectarse a la API de Gmail
        # 3. Obtener correos
        # 4. Resumirlos con Gemini
        # 5. Devolver el payload
        summary_text = f"Resumen para el usuario {user_id}: 5 correos importantes encontrados sobre {parameters.get('client_name', 'varios temas')}."
        return {"action": "summarize_inbox", "payload": {"summary": summary_text}}
    
    return intent # Por ahora, devolvemos la intención detectada


# --- OAUTH2 PARA CONEXIÓN DE CUENTAS EXTERNAS ---

# Credenciales OAuth
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET")
# IMPORTANTE: Esta debe ser la URL de TU backend en Vercel
REDIRECT_URI_GOOGLE = "https://agent-flow-backend-drab.vercel.app/google/callback" 

@app.get("/auth/google")
async def auth_google(request: Request):
    # El 'state' es crucial para la seguridad y para saber qué usuario inició el flujo
    # Asumimos que el token de Firebase se pasa como parámetro de consulta
    id_token = request.query_params.get("token")
    if not id_token:
        raise HTTPException(status_code=400, detail="Falta el token de usuario para iniciar el flujo OAuth.")
    
    scope = "https://www.googleapis.com/auth/gmail.readonly"
    url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        f"?client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={REDIRECT_URI_GOOGLE}"
        f"&response_type=code"
        f"&scope={scope}"
        f"&access_type=offline"
        f"&prompt=consent" # Fuerza a que siempre pida el consentimiento y devuelva un refresh_token
        f"&state={id_token}" # Pasamos el token del usuario como state
    )
    return RedirectResponse(url=url)


@app.get("/google/callback")
async def google_callback(request: Request):
    id_token = request.query_params.get("state")
    code = request.query_params.get("code")

    try:
        # Verificamos el token para saber a qué usuario pertenecen las credenciales
        decoded_token = auth.verify_id_token(id_token)
        user_id = decoded_token["uid"]

        # Intercambiamos el código por tokens de acceso y refresh
        token_url = "https://oauth2.googleapis.com/token"
        data = {
            "client_id": GOOGLE_CLIENT_ID,
            "client_secret": GOOGLE_CLIENT_SECRET,
            "code": code,
            "redirect_uri": REDIRECT_URI_GOOGLE,
            "grant_type": "authorization_code"
        }
        r = requests.post(token_url, data=data)
        tokens = r.json()
        
        # Guardamos los tokens de forma segura en Firestore
        if db:
            user_ref = db.collection("users").document(user_id)
            user_ref.collection("connected_accounts").document("google").set({
                "access_token": tokens.get("access_token"),
                "refresh_token": tokens.get("refresh_token"),
                "scope": tokens.get("scope"),
                "token_type": tokens.get("token_type"),
                "expiry_date": tokens.get("expires_in")
            })

        # Idealmente, redirigir a una página de éxito o usar un "deep link" a la app.
        return JSONResponse(content={"status": "Cuenta de Google conectada con éxito."})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"No se pudo vincular la cuenta: {e}")