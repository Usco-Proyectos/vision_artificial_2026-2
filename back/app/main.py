"""
main.py — API FastAPI para el Orquestador de Somnolencia.

Ejecutar (UN solo worker, el estado vive en memoria):
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import base64
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .models_loader import DEVICE, cargar_modelos
from .pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info(" Iniciando carga de modelos...")
    app.state.pipeline = Pipeline(cargar_modelos())
    logging.info(" Calentando motores (warm-up)...")
    app.state.pipeline.calentar()
    logging.info(" Servidor listo para recibir frames.")
    yield
    logging.info(" Cerrando pipeline y liberando recursos...")
    app.state.pipeline.cerrar()


app = FastAPI(
    title="Orquestador de Somnolencia",
    description="Backend de visión por computadora para detección de somnolencia en tiempo real.",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class FrameB64(BaseModel):
    imagen: str


@app.post("/frame", tags=["Inferencia"])
async def frame_binario(request: Request):
    """Recibe un frame JPEG crudo (recomendado para menor latencia)."""
    data = await request.body()
    res = await run_in_threadpool(request.app.state.pipeline.procesar_bytes, data)
    if res is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen JPEG")
    return res


@app.post("/frame/base64", tags=["Inferencia"])
async def frame_base64(body: FrameB64, request: Request):
    """Recibe un frame en formato base64 (útil para pruebas web)."""
    s = body.imagen.split(",", 1)[1] if body.imagen.startswith("data:") else body.imagen
    try:
        data = base64.b64decode(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Formato Base64 inválido")
    
    res = await run_in_threadpool(request.app.state.pipeline.procesar_bytes, data)
    if res is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen")
    return res


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Streaming en tiempo real vía WebSockets (menor overhead que POST por frame)."""
    await websocket.accept()
    pipe: Pipeline = websocket.app.state.pipeline
    try:
        while True:
            data = await websocket.receive_bytes()
            res = await run_in_threadpool(pipe.procesar_bytes, data)
            await websocket.send_json(res if res else {"error": "frame inválido o sin rostro"})
    except WebSocketDisconnect:
        logging.info("Cliente de WebSocket desconectado.")


@app.post("/reset", tags=["Administración"])
async def reset(request: Request):
    """Reinicia los contadores de frames y buffers temporales."""
    request.app.state.pipeline.reiniciar()
    return {"ok": True, "mensaje": "Pipeline reiniciado"}


@app.get("/salud", tags=["Administración"])
async def salud():
    """Health check: verifica que el servidor está vivo y en qué dispositivo corre."""
    return {
        "estado": "ok",
        "dispositivo": str(DEVICE),
        "mensaje": "Backend de somnolencia funcionando correctamente"
    }