# agent.py
import os
import time
from typing import Dict, Union

from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("AGENT_API_KEY")
NOMIC_API_KEY = os.getenv("NOMIC_API_KEY")

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from jose import JWTError, jwt
from langchain.agents import AgentType, initialize_agent, tool
from langchain.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import Tool

# If you have the official Chroma integration:
from langchain.vectorstores import Chroma

# Otherwise keep: from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_groq import ChatGroq
from langchain_nomic.embeddings import NomicEmbeddings
from pydantic import BaseModel
from starlette.middleware.base import BaseHTTPMiddleware

# ────────────────────────── 1. ENV y CREDENCIALES ──────────────────────────
load_dotenv()
API_KEY = os.getenv("AGENT_API_KEY")
JWT_SECRET = os.getenv("JWT_SECRET")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")

# ─────────────────── 2. LLM (Groq) y Embeddings (Nomic) ────────────────────
llm = ChatGroq(
    temperature=0,
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)
embeddings = NomicEmbeddings(
    model="nomic-embed-text-v1.5",
    nomic_api_key=NOMIC_API_KEY,
)

# ──────────────────── 3. TXT → VectorStore (Chroma) ────────────────────────
txt_paths = [f"{i}.txt" for i in range(1, 15)]


def extract_text_from_txt(path: str) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


# load all text
raw_text = "\n".join(extract_text_from_txt(p) for p in txt_paths)

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

...
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
    return llm.invoke(prompt_text).content


@tool
def resolver_ejercicios(input: Union[str, Dict[str, str]]) -> str:
    if isinstance(input, str):
        enunciado, contexto = input, ""
    else:
        enunciado = input.get("pregunta", "")
        contexto = input.get("contexto", "")

    adaptation_prompt = PromptTemplate.from_template(
        """
Eres un asistente virtual académico de la UNIVERSIDAD FRANCISCO DE PAULA SANTANDER, especializado en Bases de Datos y desarrollo SQL.

Has recibido un ejercicio práctico que requiere una solución técnica completa, además de una explicación clara y pedagógica de cómo se resolvió.

### Enunciado del ejercicio:
{enunciado}

### Contexto adicional:
{contexto}

### Solución paso a paso:
"""
    )
    prompt_text = adaptation_prompt.format(enunciado=enunciado, contexto=contexto)
    return llm.invoke(prompt_text)


tools = [
    Tool(
        name="consultar_conocimiento",
        func=consultar_conocimiento,
        description="Responde preguntas sobre primeros auxilios emocionales usando la base de datos.",
    ),
    Tool(
        name="resolver_ejercicios",
        func=resolver_ejercicios,
        description="Resuelve ejercicios de SQL y modelado de datos paso a paso.",
    ),
]

# ─────────────────────── 5. Inicializar Agente ────────────────────────────
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
Eres un asistente psicológico altamente capacitado en primeros auxilios emocionales.
...
Utiliza la herramienta 'consultar_conocimiento' cuando necesites recuperar info clínica.
""",
        ),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
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


request_timestamps: Dict[str, list] = {}


class SecurityMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        ip = request.client.host if request.client else "unknown"
        now = time.time()
        request_timestamps.setdefault(ip, [])
        # only keep last 60s
        request_timestamps[ip] = [t for t in request_timestamps[ip] if now - t < 60]
        if len(request_timestamps[ip]) >= 5:
            return JSONResponse(
                status_code=429, content={"error": "Demasiadas solicitudes."}
            )
        request_timestamps[ip].append(now)

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
            token_jwt = auth.removeprefix("Bearer ")
            try:
                if JWT_SECRET is None:
                    return JSONResponse(
                        status_code=500,
                        content={
                            "error": "JWT_SECRET no está configurado en el entorno."
                        },
                    )
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
    allow_origins=["http://localhost:3000", "https://aasmc.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app_fastapi.post("/token")
def generar_token(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="API Key inválida")
    if JWT_SECRET is None:
        raise HTTPException(
            status_code=500, detail="JWT_SECRET no está configurado en el entorno."
        )
    token = jwt.encode(
        {"sub": "usuario_autenticado"}, JWT_SECRET, algorithm=JWT_ALGORITHM
    )
    return {"access_token": token, "token_type": "bearer"}


@app_fastapi.post("/consultar")
def endpoint_consultar(pregunta: Pregunta):
    try:
        entrada = {
            "pregunta": pregunta.pregunta,
            "contexto": pregunta.contexto,
        }
        respuesta = agent.invoke(entrada)
        if isinstance(respuesta, dict):
            respuesta = respuesta.get("output") or str(respuesta)
        return PlainTextResponse(content=str(respuesta).strip())
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8100))
    uvicorn.run("agent:app_fastapi", host="0.0.0.0", port=port, reload=True)
