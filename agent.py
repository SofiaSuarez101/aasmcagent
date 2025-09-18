# agent.py
import os
import time

from dotenv import load_dotenv
from pathlib import Path
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from jose import JWTError, jwt
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# OpenAI (Azure) and Agents SDK
# Keep AzureOpenAI client for compatibility if needed
# Removed Agents SDK usage to avoid non-Azure tracing and simplify flow

# RAG stack
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from typing import Optional
import requests

try:
    from langchain_openai import AzureOpenAIEmbeddings  # type: ignore
except Exception:
    AzureOpenAIEmbeddings = None  # type: ignore
from langchain_nomic.embeddings import NomicEmbeddings
import fitz  # PyMuPDF

# ────────────────────────── 1. ENV y CREDENCIALES ──────────────────────────
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# Ensure OpenAI platform client does not attempt tracing with Azure key
for _var in ("OPENAI_API_KEY", "OPENAI_ORG_ID", "OPENAI_PROJECT", "OPENAI_BASE_URL"):
    if os.environ.get(_var):
        os.environ.pop(_var, None)
JWT_SECRET = os.getenv("JWT_SECRET") or ""
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ─────────────────── 2. LLM (Azure OpenAI) y Embeddings (Nomic) ────────────
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") or ""
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY") or ""
# Prefer stable GA version by default; allow env override
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")

if not AZURE_ENDPOINT or not AZURE_API_KEY:
    raise RuntimeError(
        "Faltan variables de entorno de Azure OpenAI (endpoint o api key)"
    )

AZURE_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT") or ""
# Only include temperature if explicitly provided; some models only support default (1)
_TEMP_RAW = os.getenv("AZURE_TEMPERATURE")
TEMPERATURE: Optional[float] = None
if _TEMP_RAW is not None and _TEMP_RAW.strip() != "":
    try:
        TEMPERATURE = float(_TEMP_RAW)
    except Exception:
        TEMPERATURE = None

# Prefer Azure OpenAI embeddings if configured; fallback to Nomic
AZURE_EMBEDDINGS_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDINGS_DEPLOYMENT") or ""
embeddings = None
if AzureOpenAIEmbeddings and AZURE_EMBEDDINGS_DEPLOYMENT:
    try:
        embeddings = AzureOpenAIEmbeddings(
            azure_endpoint=AZURE_ENDPOINT,
            api_key=AZURE_API_KEY,
            openai_api_version=AZURE_API_VERSION,
            azure_deployment=AZURE_EMBEDDINGS_DEPLOYMENT,
        )
    except Exception:
        embeddings = None
if embeddings is None:
    embeddings = NomicEmbeddings(model="nomic-embed-text-v1.5")


def azure_chat_complete(prompt: str) -> str:
    if not AZURE_DEPLOYMENT:
        raise RuntimeError("Falta AZURE_OPENAI_DEPLOYMENT en variables de entorno.")

    original_endpoint = AZURE_ENDPOINT.strip().rstrip("/")

    # Generate normalized endpoint host: {resource}.openai.azure.com
    normalized_endpoint = original_endpoint
    try:
        if ".cognitiveservices.azure.com" in original_endpoint:
            from urllib.parse import urlparse

            parsed = urlparse(original_endpoint)
            resource = (parsed.hostname or "").split(".")[0]
            if resource:
                normalized_endpoint = f"{parsed.scheme}://{resource}.openai.azure.com"
    except Exception:
        normalized_endpoint = original_endpoint

    endpoints_to_try = [e for e in {original_endpoint, normalized_endpoint} if e]

    headers = {"api-key": AZURE_API_KEY, "Content-Type": "application/json"}

    def _do_request(endpoint: str, api_version: str, include_temperature: bool = True):
        url = f"{endpoint}/openai/deployments/{AZURE_DEPLOYMENT}/chat/completions"
        params = {"api-version": api_version}
        body: dict[str, object] = {
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Eres un asistente psicológico altamente capacitado en primeros auxilios emocionales. "
                        "Analiza el contexto y ofrece técnicas viables en el entorno del usuario."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        }
        if include_temperature and TEMPERATURE is not None:
            body["temperature"] = TEMPERATURE
        return requests.post(url, params=params, headers=headers, json=body, timeout=60)

    errors = []
    for ep in endpoints_to_try:
        # Try configured API version first
        resp = _do_request(ep, AZURE_API_VERSION)
        if resp.status_code == 400 and AZURE_API_VERSION != "2024-10-21":
            # Retry once with stable GA version
            resp = _do_request(ep, "2024-10-21")

        # If temperature is not supported by this model, retry once without sending it
        if resp.status_code == 400:
            try:
                err_json = resp.json()
                param = (err_json.get("error") or {}).get("param") or (
                    err_json.get("innererror") or {}
                ).get("param")
            except Exception:
                param = None
            if param == "temperature":
                resp = _do_request(ep, AZURE_API_VERSION, include_temperature=False)
                if resp.status_code == 400 and AZURE_API_VERSION != "2024-10-21":
                    resp = _do_request(ep, "2024-10-21", include_temperature=False)

        if resp.ok:
            data = resp.json()
            choices = data.get("choices") or []
            if not choices:
                return ""
            msg = choices[0].get("message") or {}
            return (msg.get("content") or "").strip()

        try:
            err = resp.json()
        except Exception:
            err = {"text": resp.text}
        errors.append({"endpoint": ep, "status": resp.status_code, "error": err})

    raise HTTPException(status_code=500, detail={"azure_errors": errors})


# ───────────────────── 3. PDFs → VectorStore (Chroma) ──────────────────────
pdf_paths = [r"1.pdf", r"3.pdf"]


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    return text


raw_text = "\n".join(extract_text_from_pdf(p) for p in pdf_paths)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
documents = [
    Document(page_content=chunk) for chunk in text_splitter.split_text(raw_text)
]
vectorstore = Chroma.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()


"""
4. Herramientas (function_tool) para el agente usando OpenAI Agents SDK.
Implementa la misma lógica RAG: recuperar y adaptar la respuesta con el LLM (Azure OpenAI).
"""


def _adaptacion_prompt(respuesta_original: str, contexto: str) -> str:
    return f"""
Eres un agente psicológico de la UNIVERSIDAD FRANCISCO DE PAULA SANTANDER especializado en primeros auxilios emocionales.

Has recibido una respuesta basada en el conocimiento clínico de una base de datos validada. Tu tarea es adaptar esa respuesta al contexto específico de la pregunta del usuario, sin alterar su contenido esencial ni distorsionar la información recuperada.

- Si el usuario se encuentra en el trabajo, en clase o en un espacio público, ajusta la forma de aplicar las técnicas para que sean viables y discretas.
- Nunca elimines el conocimiento clave, solo reformula lo necesario de forma práctica y accesible.
- Muestra empatía, cercanía y comprensión emocional.
- No realices diagnósticos clínicos; tu función es contener, orientar y sugerir estrategias útiles.
- Mantén la respuesta completamente en Español.
- Si recomiendas técnicas, explica en qué consisten y da ejemplos.


{respuesta_original}

### Contexto adicional:
{contexto}

Redacta la respuesta final breve, clara y cálida:
"""


def consultar_conocimiento(pregunta: str, contexto: str = "") -> str:
    """Consulta el flujo de análisis psicológico (RAG) y adapta la respuesta al contexto."""
    docs = retriever.invoke(pregunta)
    doc_text = (
        "\n\n".join(d.page_content for d in docs[:2])
        or "No se encontró información relevante."
    )

    prompt = _adaptacion_prompt(doc_text.strip(), contexto)
    return azure_chat_complete(prompt)


INSTRUCTIONS = """
    Eres un asistente psicológico altamente capacitado en primeros auxilios emocionales.
    Objetivo: acompañar en momentos de malestar emocional, estrés o ansiedad.
    - Adapta las recomendaciones al entorno del usuario.
    - Ofrece técnicas prácticas y beneficios esperados.
    - Responde siempre en Español, con calidez y claridad.
    """


# ─────────────────────────── 6. FastAPI y seguridad ────────────────────────
class Pregunta(BaseModel):
    pregunta: str
    contexto: str = ""


request_timestamps = {}


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Rate-limit sencillo por IP
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        request_timestamps.setdefault(ip, [])
        request_timestamps[ip] = [t for t in request_timestamps[ip] if now - t < 60]
        if len(request_timestamps[ip]) >= 5:
            return JSONResponse(
                status_code=429,
                content={"error": "Demasiadas solicitudes. Intente más tarde."},
            )

        request_timestamps[ip].append(now)

        # Protección de /consultar (solo JWT)
        if request.url.path == "/consultar":
            auth = request.headers.get("Authorization", "")
            if not auth.startswith("Bearer "):
                return JSONResponse(
                    status_code=401, content={"error": "Token JWT faltante o inválido."}
                )

            token_jwt = auth.replace("Bearer ", "")
            try:
                jwt.decode(token_jwt, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            except JWTError:
                return JSONResponse(
                    status_code=403, content={"error": "Token JWT inválido o expirado."}
                )

        return await call_next(request)


app_fastapi = FastAPI()
app = app_fastapi
app_fastapi.add_middleware(SecurityMiddleware)

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:3001",
        "https://aasmc.vercel.app",
        "https://aasmcv2.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────── Endpoints ─────────────────────────────────────────
@app_fastapi.get("/health")
async def health():
    return {"status": "ok"}


@app_fastapi.post("/token")
async def generar_token():
    token = jwt.encode(
        {"sub": "usuario_autenticado"}, JWT_SECRET, algorithm=JWT_ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}


@app_fastapi.post("/consultar")
async def endpoint_consultar(
    pregunta: Pregunta,
    request: Request,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    try:
        # Very simple risk detection
        texto = (pregunta.pregunta or "").lower()
        keywords_alta = [
            "suicid",
            "matarme",
            "quitarme la vida",
            "no quiero vivir",
            "muy triste",
            "depres",
            "desesperado",
            "autoagredir",
            "autolesion",
        ]
        is_alert = any(k in texto for k in keywords_alta)

        salida = consultar_conocimiento(pregunta.pregunta, pregunta.contexto)

        # Determine user id from Authorization JWT (preferred) or header fallback
        uid: int | None = None
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token_jwt = auth.replace("Bearer ", "")
            try:
                payload = jwt.decode(token_jwt, JWT_SECRET, algorithms=[JWT_ALGORITHM])
                sub = payload.get("sub")
                if isinstance(sub, (int, float)):
                    uid = int(sub)
                elif isinstance(sub, str) and sub.isdigit():
                    uid = int(sub)
            except JWTError:
                uid = None
        if uid is None and x_user_id:
            try:
                uid = int(x_user_id)
            except ValueError:
                uid = None

        # If alert and we have a user id, persist alerta in backend
        if is_alert and uid is not None:
            try:
                backend = os.getenv("BACKEND_URL", "http://localhost:8000")
                payload = {
                    "id_estudiante": uid,
                    "texto": pregunta.pregunta,
                    "severidad": "CRITICA",
                }
                requests.post(
                    f"{backend.rstrip('/')}/alertas/",
                    json=payload,
                    timeout=10,
                )
            except Exception:
                pass

        return PlainTextResponse(content=str(salida).strip())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ──────────────────────────── Ejecutar local ───────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port_str = os.getenv("PORT") or "8100"
    port = int(port_str)
    uvicorn.run("agent:app", host="0.0.0.0", port=port, reload=True)
