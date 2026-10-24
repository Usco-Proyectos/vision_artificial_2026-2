"""
pipeline.py — Orquestación por frame: extracción → M1/M2 (cada frame) → buffer 3 FPS (M3/M4) → cascada.
CORREGIDO: M3 es modo señales (vectores de 6 valores), M4 usa 4 features exactos.
"""
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import config as C
from .inferencia import predecir_m1, predecir_m2, predecir_m3, predecir_m4
from .models_loader import Modelos
from .orchestrator import CascadaSeguridad, Umbrales
from .vision import ExtractorRostro, descriptores_ventana_m4, preparar


class MedidorFPS:
    """FPS reales de procesamiento = frames completados / tiempo, sobre los últimos N."""
    def __init__(self, n: int = 30):
        self._t = deque(maxlen=n)

    def tick(self) -> float:
        self._t.append(time.perf_counter())
        if len(self._t) < 2:
            return 0.0
        return (len(self._t) - 1) / (self._t[-1] - self._t[0] + 1e-9)


class Pipeline:
    def __init__(self, modelos: Modelos):
        self.mod = modelos
        self._lock = threading.Lock()       # torch + mediapipe: un frame a la vez, en orden
        self.extractor = ExtractorRostro()
        self.cascada = CascadaSeguridad(Umbrales())
        self.fps = MedidorFPS()
        
        # Últimas probabilidades de M1/M2 (para construir señales de M3)
        self._ultima_m1_der = None
        self._ultima_m1_izq = None
        self._ultima_m2 = None
        
        self.reiniciar()

    def reiniciar(self):
        with self._lock:
            self.n_frames = 0
            # Buffer de SEÑALES para M3: vectores de 6 valores o None
            self.buf_senales = deque(maxlen=C.VENTANA_M3)
            # Buffer de descriptores para M4: (ear_der, ear_izq, mar) o None
            self.buf_desc = deque(maxlen=C.VENTANA_M3)
            self.m3_estado, self.m3_prob = "sin_ventana", None
            self.cascada.reiniciar()
            self._ultima_m1_der = None
            self._ultima_m1_izq = None
            self._ultima_m2 = None

    def cerrar(self):
        self.extractor.cerrar()

    def calentar(self):
        """Forward de relleno para evitar el lag del primer frame (cudnn/allocs)."""
        try:
            z_m1 = np.zeros((3, *C.TAM_M1[::-1]), np.float32)
            predecir_m1(self.mod.m1, [z_m1, z_m1])
            predecir_m2(self.mod.m2, np.zeros((3, *C.TAM_M2[::-1]), np.float32))
            
            # M3: tensor (1, 10, 6) con NaN al final (modo señales)
            z_m3 = np.zeros((1, C.VENTANA_M3, C.DIM_SENALES_M3), dtype=np.float32)
            z_m3[0, 5:, :] = float('nan')  # últimos 5 frames son NaN
            predecir_m3(self.mod.m3, z_m3)
            
            # M4: DataFrame con 4 features exactos
            df_m4 = pd.DataFrame([[0.3, 0.1, 2, 0.4]], columns=C.FEATURES_M4)
            predecir_m4(self.mod.m4, df_m4)
        except Exception as e:
            print(f"[warm-up] aviso: {e}")

    def _construir_senal(self, ext) -> Optional[np.ndarray]:
        """
        Construye el vector de 6 señales para M3:
        [p_ojo_der, p_ojo_izq, p_boca, ear_der, ear_izq, mar]
        """
        if ext is None:
            return None
        
        if self._ultima_m1_der is None or self._ultima_m1_izq is None or self._ultima_m2 is None:
            return None
        
        return np.array([
            self._ultima_m1_der,
            self._ultima_m1_izq,
            self._ultima_m2,
            ext.ear_der,
            ext.ear_izq,
            ext.mar
        ], dtype=np.float32)

    def _muestrear(self, ext) -> tuple:
        """
        Muestrea cada STRIDE_M3 frames. Construye la señal para M3 y descriptores para M4.
        Devuelve (m3_pred, m4_pred) o (None, None) si no hay datos suficientes.
        """
        senal = self._construir_senal(ext)
        self.buf_senales.append(senal)
        
        if ext is not None:
            self.buf_desc.append((ext.ear_der, ext.ear_izq, ext.mar))
        else:
            self.buf_desc.append(None)
        
        if len(self.buf_senales) < C.VENTANA_M3:
            self.m3_estado = "sin_ventana"
            return None, None
        
        validos = sum(1 for s in self.buf_senales if s is not None)
        if validos < C.MIN_FRAMES_VALIDOS_M3:
            self.m3_estado, self.m3_prob = "insuficiente", None
            return 0, 0
        
        # Construir tensor para M3: (1, 10, 6) con NaN donde faltan frames
        senales_array = []
        for s in self.buf_senales:
            if s is not None:
                senales_array.append(s)
            else:
                senales_array.append(np.full(C.DIM_SENALES_M3, np.nan, dtype=np.float32))
        
        tensor_m3 = np.stack(senales_array, axis=0)  # (10, 6)
        tensor_m3 = tensor_m3[np.newaxis, ...]       # (1, 10, 6)
        
        self.m3_prob = predecir_m3(self.mod.m3, tensor_m3)
        self.m3_estado = "ok"
        m3_pred = int(self.m3_prob >= C.UMBRAL_PROB_M3)
        
        # Construir features para M4
        desc_validos = [d for d in self.buf_desc if d is not None]
        if len(desc_validos) < C.MIN_FRAMES_VALIDOS_M3:
            return m3_pred, 0
        
        ear_der_seq = [d[0] for d in desc_validos]
        ear_izq_seq = [d[1] for d in desc_validos]
        mar_seq = [d[2] for d in desc_validos]
        
        features_m4 = descriptores_ventana_m4(ear_der_seq, ear_izq_seq, mar_seq)
        
        # M4 espera un DataFrame con nombres de columnas
        df_m4 = pd.DataFrame([features_m4], columns=C.FEATURES_M4)
        m4_pred = predecir_m4(self.mod.m4, df_m4)
        
        return m3_pred, m4_pred

    def procesar_bytes(self, data: bytes) -> Optional[dict]:
        frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return None if frame is None else self.procesar(frame)

    def procesar(self, frame: np.ndarray) -> dict:
        with self._lock:
            t0 = time.perf_counter()
            self.n_frames += 1
            tm = {}

            ext = self.extractor.procesar(frame)
            tm["extraccion"] = time.perf_counter()

            m1_prob = m2_prob = None
            if ext:
                m1_prob = predecir_m1(self.mod.m1, [
                    preparar(ext.ojo_der, C.TAM_M1, C.M1_ESCALA_GRISES),
                    preparar(ext.ojo_izq, C.TAM_M1, C.M1_ESCALA_GRISES)
                ])
                m2_prob = predecir_m2(self.mod.m2, preparar(ext.boca, C.TAM_M2, C.M2_ESCALA_GRISES))
                
                self._ultima_m1_der = m1_prob[0]
                self._ultima_m1_izq = m1_prob[1]
                self._ultima_m2 = m2_prob
            
            tm["m1_m2"] = time.perf_counter()

            m3_pred = m4_pred = None
            if (self.n_frames - 1) % C.STRIDE_M3 == 0:
                m3_pred, m4_pred = self._muestrear(ext)
            tm["m3_m4"] = time.perf_counter()

            v = self.cascada.step(m1_prob, m2_prob, m3_pred, m4_pred)
            fps = self.fps.tick()
            t_fin = time.perf_counter()

            ms = lambda a, b: round((a - b) * 1000, 2)
            return {
                "frame": self.n_frames,
                "veredicto": v.veredicto,
                "regla_activada": v.regla,
                "rostro_detectado": ext is not None,
                "modelos": {
                    "m1": {
                        "prob_der": self._ultima_m1_der,
                        "prob_izq": self._ultima_m1_izq,
                        "ojos_cerrados_frame": v.m1_cerrado_frame,
                        "frames_seguidos": v.m1_frames_seguidos,
                        "umbral_frames": self.cascada.u.frames_ojos,
                        "activo": v.m1_sostenido
                    },
                    "m2": {
                        "prob": m2_prob,
                        "bostezo": v.m2_bostezo
                    },
                    "m3": {
                        "estado": self.m3_estado,
                        "prob": self.m3_prob,
                        "prediccion": v.m3,
                        "frames_en_ventana": len(self.buf_senales),
                        "frames_validos": sum(1 for s in self.buf_senales if s is not None)
                    },
                    "m4": {
                        "prediccion": v.m4
                    },
                },
                "rendimiento": {
                    "fps_backend": round(fps, 1),
                    "latencia_ms": ms(t_fin, t0),
                    "tiempos_ms": {
                        "extraccion": ms(tm["extraccion"], t0),
                        "m1_m2": ms(tm["m1_m2"], tm["extraccion"]),
                        "m3_m4": ms(tm["m3_m4"], tm["m1_m2"])
                    },
                },
            }