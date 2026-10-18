"""Detección de rostro con MediaPipe, recortes de ojos/boca/rostro y EAR/MAR.

Se usa en el procesamiento de YawDD y UTA-RLDD, y después en la aplicación,
para que el preprocesamiento sea exactamente el mismo al entrenar y al usar el modelo.
"""
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

import config

# Índices de la malla facial de MediaPipe (468 puntos) de uso común para EAR y MAR.
# Orden EAR: p1 y p4 son las esquinas; (p2, p6) y (p3, p5) los pares verticales.
# "der"/"izq" se refieren al ojo derecho/izquierdo de la persona.
OJO_DER = [33, 160, 158, 133, 153, 144]
OJO_IZQ = [362, 385, 387, 263, 373, 380]
# MAR: tres pares verticales de los labios internos y las comisuras.
BOCA_VERT = [(81, 178), (13, 14), (311, 402)]
BOCA_HORIZ = (61, 291)

# Contornos para recortar cada región
CONTORNO_OJO_DER = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
CONTORNO_OJO_IZQ = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
CONTORNO_BOCA = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]


@dataclass
class Rostro:
    ojo_der: np.ndarray
    ojo_izq: np.ndarray
    boca: np.ndarray
    rostro: np.ndarray
    ear_der: float
    ear_izq: float
    mar: float


def _dist(a, b):
    return float(np.linalg.norm(a - b))


def ear(p, idx):
    p1, p2, p3, p4, p5, p6 = (p[i] for i in idx)
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * _dist(p1, p4) + 1e-6)


def mar(p):
    vert = sum(_dist(p[a], p[b]) for a, b in BOCA_VERT)
    return vert / (len(BOCA_VERT) * _dist(p[BOCA_HORIZ[0]], p[BOCA_HORIZ[1]]) + 1e-6)


def recorte_cuadrado(gris, puntos, margen, tam):
    """Recorta un cuadrado centrado en los puntos, con un margen relativo."""
    x0, y0 = puntos.min(axis=0)
    x1, y1 = puntos.max(axis=0)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    lado = max(x1 - x0, y1 - y0) * (1 + 2 * margen)
    h, w = gris.shape
    a, b = int(max(cx - lado / 2, 0)), int(min(cx + lado / 2, w))
    c, d = int(max(cy - lado / 2, 0)), int(min(cy + lado / 2, h))
    if b - a < 4 or d - c < 4:
        return None
    return cv2.resize(gris[c:d, a:b], (tam, tam), interpolation=cv2.INTER_AREA)


def enderezar(gris, p):
    """Gira la imagen (y los puntos) para que la línea entre los ojos quede horizontal.

    Algunos videos de UTA-RLDD están grabados de lado (girados 90°); sin esto los
    recortes salen girados. También corrige cabezas inclinadas.
    """
    der, izq = p[CONTORNO_OJO_DER].mean(axis=0), p[CONTORNO_OJO_IZQ].mean(axis=0)
    dx, dy = izq - der  # en una cara derecha el ojo derecho de la persona queda a la izquierda
    angulo = np.degrees(np.arctan2(dy, dx))
    if abs(angulo) < 1:
        return gris, p
    h, w = gris.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angulo, 1.0)
    # Lienzo más grande para que no se corte nada al girar
    cos, sen = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sen + w * cos), int(h * cos + w * sen)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    girada = cv2.warpAffine(gris, M, (nw, nh), flags=cv2.INTER_LINEAR)
    puntos = np.hstack([p, np.ones((len(p), 1), np.float32)]) @ M.T
    return girada, puntos.astype(np.float32)


class DetectorRostro:
    def __init__(self, modo_video=False):
        modo = vision.RunningMode.VIDEO if modo_video else vision.RunningMode.IMAGE
        opciones = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(config.MODELO_LANDMARKS)),
            running_mode=modo,
            num_faces=1,
        )
        self.modo_video = modo_video
        self.detector = vision.FaceLandmarker.create_from_options(opciones)

    def procesar(self, bgr, tiempo_ms=None):
        """Devuelve un Rostro o None si no se detectó cara o un recorte falló."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        imagen = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self.modo_video:
            res = self.detector.detect_for_video(imagen, int(tiempo_ms))
        else:
            res = self.detector.detect(imagen)
        if not res.face_landmarks:
            return None
        h, w = bgr.shape[:2]
        p = np.array([(l.x * w, l.y * h) for l in res.face_landmarks[0]], dtype=np.float32)
        gris, p = enderezar(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), p)
        recortes = (
            recorte_cuadrado(gris, p[CONTORNO_OJO_DER], 0.4, config.TAM_OJO),
            recorte_cuadrado(gris, p[CONTORNO_OJO_IZQ], 0.4, config.TAM_OJO),
            recorte_cuadrado(gris, p[CONTORNO_BOCA], 0.3, config.TAM_BOCA),
            recorte_cuadrado(gris, p, 0.1, config.TAM_ROSTRO),
        )
        if any(r is None for r in recortes):
            return None
        return Rostro(*recortes, ear_der=ear(p, OJO_DER), ear_izq=ear(p, OJO_IZQ), mar=mar(p))

    def cerrar(self):
        self.detector.close()
