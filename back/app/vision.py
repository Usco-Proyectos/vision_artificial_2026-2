"""
vision.py — Segmentación facial con MediaPipe Face Mesh y extracción de entradas:
  ojos → M1 | boca → M2 | EAR/MAR → M3 (señales) y M4 (descriptores)
"""
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# ⭐ IMPORTACIÓN ESTÁNDAR Y OFICIAL (Pylance la reconoce y funciona en runtime)
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

from . import config as C

# Índices Face Mesh (468 puntos). Orden EAR: [esquina_ext, sup1, sup2, esquina_int, inf1, inf2]
IDX_OJO_DER = [33, 160, 158, 133, 153, 144]   # ojo DERECHO del sujeto
IDX_OJO_IZQ = [362, 385, 387, 263, 373, 380]  # ojo IZQUIERDO del sujeto
IDX_LABIO_EXT = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
                 185, 40, 39, 37, 0, 267, 269, 270, 409]
# Los mismos puntos del entrenamiento (src/dataset/rostro.py, ec. 6 del artículo)
MAR_VERTICALES = [(81, 178), (13, 14), (311, 402)]
MAR_HORIZONTAL = (61, 291)

_MEAN = np.array(C.MEDIA, np.float32)
_STD = np.array(C.STD, np.float32)

@dataclass
class Extraccion:
    """Resultados de procesar UN frame. Ya NO incluye 'rostro' (M3 es señales)."""
    ojo_der: np.ndarray      # recorte BGR del ojo derecho del sujeto
    ojo_izq: np.ndarray      # recorte BGR del ojo izquierdo del sujeto
    boca: np.ndarray         # recorte BGR de la boca
    ear_der: float           # EAR del ojo derecho (ya calibrado)
    ear_izq: float           # EAR del ojo izquierdo (ya calibrado)
    mar: float               # MAR de la boca (ya calibrado)


def _d(a, b) -> float:
    """Distancia euclidiana entre dos puntos."""
    return float(np.linalg.norm(a - b))


def calcular_ear(p: np.ndarray) -> float:
    """EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||). Fórmula de Soukupová & Čech."""
    return (_d(p[1], p[5]) + _d(p[2], p[4])) / (2.0 * _d(p[0], p[3]) + 1e-6)


def calcular_mar(pts: np.ndarray) -> float:
    """MAR = suma(distancias verticales) / (3 * distancia horizontal)."""
    v = sum(_d(pts[a], pts[b]) for a, b in MAR_VERTICALES)
    return v / (3.0 * _d(pts[MAR_HORIZONTAL[0]], pts[MAR_HORIZONTAL[1]]) + 1e-6)


def recortar(img: np.ndarray, pts: np.ndarray, margen: float) -> Optional[np.ndarray]:
    """Recorte cuadrado alrededor de un conjunto de puntos (con margen relativo)."""
    H, W = img.shape[:2]
    x0, y0 = pts.min(0)
    x1, y1 = pts.max(0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    mitad = max(x1 - x0, y1 - y0) * (1 + margen) / 2
    xa, xb = int(max(0, cx - mitad)), int(min(W, cx + mitad))
    ya, yb = int(max(0, cy - mitad)), int(min(H, cy + mitad))
    if xb - xa < 4 or yb - ya < 4:
        return None
    return img[ya:yb, xa:xb]


class ExtractorRostro:
    """Un extractor = un flujo de video (MediaPipe mantiene tracking entre frames)."""

    def __init__(self):
        self._mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,       # modo video: tracking (más rápido)
            max_num_faces=1,
            refine_landmarks=False,        # sin iris: no se necesita
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def cerrar(self):
        self._mesh.close()

    def procesar(self, frame_bgr: np.ndarray) -> Optional[Extraccion]:
        """Procesa un frame BGR y devuelve la extracción, o None si no hay rostro válido."""
        h, w = frame_bgr.shape[:2]
        m = max(h, w)
        if m > C.MAX_LADO_PROCESO:                       # reducir ANTES de todo
            e = C.MAX_LADO_PROCESO / m
            frame_bgr = cv2.resize(frame_bgr, (int(w * e), int(h * e)), interpolation=cv2.INTER_AREA)
            h, w = frame_bgr.shape[:2]

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False                      # evita copia interna
        res = self._mesh.process(rgb)
        if not res.multi_face_landmarks:
            return None

        lm = res.multi_face_landmarks[0].landmark
        pts = np.array([(p.x * w, p.y * h) for p in lm], dtype=np.float32)

        # Extraer recortes (ya NO se extrae rostro, M3 es señales)
        ojo_der = recortar(frame_bgr, pts[IDX_OJO_DER], C.MARGEN_OJO)
        ojo_izq = recortar(frame_bgr, pts[IDX_OJO_IZQ], C.MARGEN_OJO)
        boca = recortar(frame_bgr, pts[IDX_LABIO_EXT], C.MARGEN_BOCA)
        if any(x is None for x in (ojo_der, ojo_izq, boca)):
            return None

        # Calcular EAR / MAR (con calibración opcional MediaPipe→UTA)
        ear_der = C.EAR_CAL_A * calcular_ear(pts[IDX_OJO_DER]) + C.EAR_CAL_B
        ear_izq = C.EAR_CAL_A * calcular_ear(pts[IDX_OJO_IZQ]) + C.EAR_CAL_B
        mar = C.MAR_CAL_A * calcular_mar(pts) + C.MAR_CAL_B

        # Opción: intercambiar ojos si MediaPipe los da cruzados
        if C.INTERCAMBIAR_OJOS:
            ojo_der, ojo_izq = ojo_izq, ojo_der
            ear_der, ear_izq = ear_izq, ear_der

        return Extraccion(ojo_der, ojo_izq, boca, ear_der, ear_izq, mar)


def preparar(img_bgr: np.ndarray, tam: tuple, grises: bool = False) -> np.ndarray:
    """
    BGR uint8 → array (3,H,W) float32 normalizado.
    Replica el preprocesamiento de entrenamiento de M1/M2:
      gris → /255 → replicar a 3 canales → Normalize(ImageNet) → CHW
    """
    img = cv2.resize(img_bgr, tam, interpolation=cv2.INTER_AREA)
    if grises:
        g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.merge([g, g, g])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def descriptores_ventana_m4(ear_der_seq, ear_izq_seq, mar_seq) -> np.ndarray:
    """
    ⭐ REPLICA EXACTAMENTE analisis/figuras_uta.py -> descriptores_ventana.
    Recibe las secuencias (un valor por frame de la ventana) y devuelve los 4 features
    de M4 en el orden EXACTO: [ear_medio, perclos, parpadeos, mar_max].

    Args:
        ear_der_seq: EAR del ojo derecho por frame (lista/array)
        ear_izq_seq: EAR del ojo izquierdo por frame (lista/array)
        mar_seq: MAR por frame (lista/array)
    Returns:
        np.ndarray de forma (4,) con los features para M4
    """
    ear_der_seq = np.asarray(ear_der_seq, dtype=np.float64)
    ear_izq_seq = np.asarray(ear_izq_seq, dtype=np.float64)
    mar_seq = np.asarray(mar_seq, dtype=np.float64)

    # EAR promedio de ambos ojos (igual que figuras_uta.py)
    ear = (ear_der_seq + ear_izq_seq) / 2.0
    cerrado = ear < C.UMBRAL_EAR_CERRADO

    # Parpadeos: transiciones abierto→cerrado + si el primer frame empieza cerrado
    parpadeos = int(np.sum(~cerrado[:-1] & cerrado[1:])) + int(cerrado[0])

    return np.array([
        ear.mean(),           # ear_medio
        cerrado.mean(),       # perclos (fracción de frames con ojo cerrado)
        parpadeos,            # parpadeos
        mar_seq.max(),        # mar_max
    ], dtype=np.float32)