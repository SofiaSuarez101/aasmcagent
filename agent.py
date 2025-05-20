# agent.py
import os
import time
from typing import Dict, Union

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from jose import JWTError, jwt
from langchain.agents import AgentType, initialize_agent, tool
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_nomic.embeddings import NomicEmbeddings
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

# ────────────────────────── 1. ENV y CREDENCIALES ──────────────────────────
load_dotenv()
print("AGENT_API_KEY BACKEND:", os.getenv("AGENT_API_KEY"))
API_KEY = os.getenv("AGENT_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ─────────────────── 2. LLM (Groq) y Embeddings (Nomic) ────────────────────
llm = ChatGroq(
    temperature=0,
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
)
embeddings = SentenceTransformerEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

# ───────────────────── 3. PDFs → VectorStore (Chroma) ──────────────────────
import fitz  # PyMuPDF

pdf_paths = [
    r"1.txt",
    r"2.txt",
    r"3.txt",
    r"4.txt",
    r"5.txt",
    r"6.txt",
    r"7.txt",
    r"8.txt",
    r"9.txt",
    r"10.txt",
    r"11.txt",
    r"12.txt",
    r"13.txt",
    r"14.txt",
]


def extract_text_from_pdf(path: str) -> str:
    doc = fitz.open(path)
    text = ""
    for page in doc:
        text += page.get_text()
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


# ─────────────────────── 4. Tools para el agente ───────────────────────────
@tool
def consultar_conocimiento(input: Union[str, Dict[str, str]]) -> str:
    """
    Consulta el flujo de análisis psicológico para responder preguntas relacionadas con ansiedad,
    relajación, meditación u otras necesidades emocionales del usuario.
    """
    if isinstance(input, str):
        pregunta, contexto = input, ""
    else:
        pregunta = input.get("pregunta", "")
        contexto = input.get("contexto", "")

    docs = retriever.invoke(pregunta)
    doc_text = (
        "\n\n".join(d.page_content for d in docs[:2])
        or "No se encontró información relevante."
    )

    adaptation_prompt = PromptTemplate.from_template(
        """
Eres un agente psicológico de la UNIVERSIDAD FRANCISCO DE PAULA SANTANDER especializado en primeros auxilios emocionales.

Has recibido una respuesta basada en el conocimiento clínico de una base de datos validada. Tu tarea es **adaptar esa respuesta al contexto específico de la pregunta del usuario**, sin alterar su contenido esencial ni distorsionar la información recuperada.

- Si el usuario se encuentra en el trabajo, en clase o en un espacio público, **ajusta la forma de aplicar las técnicas** para que sean más viables y discretas en ese entorno (por ejemplo: no sugieras acostarse o cerrar los ojos si no es posible).
- **Nunca elimines el conocimiento clave** contenido en la respuesta original, solo reformula lo que no se adapta al contexto de forma práctica y accesible.
- Muestra empatía, cercanía y comprensión emocional en el lenguaje utilizado.
- No realices diagnósticos clínicos. Tu función es contener, orientar y sugerir estrategias útiles.
- Mantén siempre la respuesta completamente en Español.
- Si recomiendas una o más técnicas especifica en qué consisten y proporciona ejemplos.
- Se suministra además el historial de la conversación como contexto adicional.


### Respuesta original:
{respuesta_original}

### Contexto adicional:
{contexto}

### Respuesta explicada:
"""
    )
    prompt_text = adaptation_prompt.format(
        respuesta_original=doc_text.strip(), contexto=contexto
    )
    return llm.invoke(prompt_text)


tools = [
    Tool(
        name="consultar_conocimiento",
        func=consultar_conocimiento,
        description="Preguntas teóricas sobre conceptos de bases de datos.",
    ),
]

# ──────────────────────────────── 5. Agente ────────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
 Eres un asistente psicológico altamente capacitado en primeros auxilios emocionales.

              Tu objetivo es acompañar al usuario en momentos de malestar emocional, estrés, ansiedad, confusión o cualquier otra necesidad psicológica urgente.

              Analiza cuidadosamente el contexto en el que se encuentra el usuario (por ejemplo: en el trabajo, en clase, en casa, en un espacio público, etc.). No hagas suposiciones sin datos.

              Identifica con claridad la necesidad emocional o informativa que expresa el usuario y proporciona respuestas que:
              - Sean empáticas, útiles y contenidas emocionalmente.
              - Estén adaptadas al entorno del usuario (por ejemplo, si está en clase, evita sugerir acostarse o cerrar los ojos).
              - Incluyan técnicas de regulación emocional o estrategias prácticas que pueda aplicar en su contexto inmediato.
              - Señalen los beneficios esperados de dichas técnicas (por ejemplo: respirar profundamente ayuda a reducir la ansiedad y recuperar la claridad mental).

              Utiliza la herramienta 'consultar' para recuperar conocimientos específicos cuando sea necesario, enviando siempre la pregunta completa del usuario como entrada.

              Antes de responder:
              1. Revisa si las recomendaciones son viables en el entorno descrito.
              2. Descarta aquellas que no se puedan realizar.
              3. Reformula la respuesta para que sea breve, clara, cálida y completamente en Español.

              Recuerda: tu rol no es diagnosticar, sino contener, orientar y ofrecer recursos concretos y seguros para el bienestar emocional inmediato del usuario.

""",
        ),
        ("user", "{input}"),
    ]
)

agent = initialize_agent(
    tools,
    llm,
    prompt=prompt,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    return_intermediate_steps=False,
    handle_parsing_errors=True,
)


# ─────────────────────────── 6. FastAPI y seguridad ────────────────────────
class Pregunta(BaseModel):
    pregunta: str
    contexto: str = ""


request_timestamps = {}


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Rate-limit sencillo por IP
        ip = request.client.host
        now = time.time()
        request_timestamps.setdefault(ip, [])
        request_timestamps[ip] = [t for t in request_timestamps[ip] if now - t < 60]
        if len(request_timestamps[ip]) >= 5:
            return JSONResponse(
                status_code=429,
                content={"error": "Demasiadas solicitudes. Intente más tarde."},
            )

        request_timestamps[ip].append(now)

        # Protección de /consultar
        if request.url.path == "/consultar":
            if request.headers.get("x-api-key") != API_KEY:
                return JSONResponse(
                    status_code=401, content={"error": "API Key inválida."}
                )

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
app_fastapi.add_middleware(SecurityMiddleware)

app_fastapi.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://aasmc.vercel.app/",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────── Endpoints ─────────────────────────────────────────
@app_fastapi.post("/token")
def generar_token(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")

    token = jwt.encode(
        {"sub": "usuario_autenticado"}, JWT_SECRET, algorithm=JWT_ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}


@app_fastapi.post("/consultar")
def endpoint_consultar(pregunta: Pregunta):
    try:
        entrada = {
            "input": {
                "pregunta": pregunta.pregunta,
                "contexto": pregunta.contexto,
            }
        }
        respuesta = agent.invoke(entrada)

        # ── Normalizar salida del agente ──────────────────────
        if isinstance(respuesta, dict):
            # LangChain a veces devuelve {"output": "..."}
            respuesta = respuesta.get("output") or str(respuesta)

        if not isinstance(respuesta, str):
            respuesta = str(respuesta)

        return PlainTextResponse(content=respuesta.strip())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ──────────────────────────── Ejecutar local ───────────────────────────────
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8100))
    uvicorn.run("agent:app_fastapi", host="0.0.0.0", port=port, reload=True)
