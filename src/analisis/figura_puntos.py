"""Figura 3 del artículo: puntos de MediaPipe que se usan para el EAR y el MAR (ecs. 5 y 6).

Saca los 468 puntos de un rostro de YawDD (ojos y boca abiertos) y dibuja SOLO los puntos,
no la foto, para no mostrar la cara de nadie.
  a) malla completa con los puntos del EAR y del MAR resaltados
  b) ojo derecho: p1..p6 con sus índices y las distancias del EAR
  c) boca: los tres pares verticales y las comisuras del MAR

Uso:  python -m analisis.figura_puntos   (desde la carpeta src)
"""
import cv2
import matplotlib
import mediapipe as mp
import numpy as np
import pandas as pd

import config
from analisis.figuras_uta import FONDO, SALIDA, TINTA, TINTA_2
from dataset.rostro import BOCA_HORIZ, BOCA_VERT, OJO_DER, OJO_IZQ, DetectorRostro, ear, mar
from mediapipe.tasks.python.vision import FaceLandmarksConnections as C

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#c9c8c2"
BOCA = [i for par in BOCA_VERT for i in par] + list(BOCA_HORIZ)


def puntos_de_un_rostro():
    """Primer recorte de rostro de YawDD con ojos abiertos y boca abierta (para que se vean las
    distancias del MAR). Los puntos se enderezan para que la línea entre los ojos quede horizontal."""
    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False, usecols=["fuente", "ruta_rostro", "mar"])
    rutas = d[(d.fuente == "yawdd") & d.ruta_rostro.notna() & (d.mar > 0.3)].ruta_rostro
    det = DetectorRostro()
    for r in rutas.sample(min(len(rutas), 300), random_state=config.SEMILLA):
        img = cv2.imread(str(config.PROCESADO / r))
        if img is None:
            continue
        img = cv2.resize(img, (480, 480), interpolation=cv2.INTER_CUBIC)  # el recorte de 160 es chico para MediaPipe
        res = det.detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
        if not res.face_landmarks:
            continue
        p = np.array([(l.x * 480, l.y * 480) for l in res.face_landmarks[0]])
        p = enderezar_puntos(p)
        # se mide sobre los mismos puntos que se van a dibujar
        if mar(p) > 0.35 and min(ear(p, OJO_DER), ear(p, OJO_IZQ)) > 0.22:
            det.cerrar()
            print(f"rostro: {r}  EAR {ear(p, OJO_DER):.2f}/{ear(p, OJO_IZQ):.2f}  MAR {mar(p):.2f}")
            return p
    det.cerrar()
    raise SystemExit("No hubo un rostro con ojos y boca abiertos en los recortes elegidos.")


def enderezar_puntos(p):
    """Gira los puntos para que la línea entre las esquinas externas de los ojos quede horizontal."""
    dx, dy = p[263] - p[33]
    a = -np.arctan2(dy, dx)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    return (p - p.mean(0)) @ R.T


def malla(ax, p, conexiones, color, ancho):
    for c in conexiones:
        ax.plot(*p[[c.start, c.end]].T, color=color, linewidth=ancho, zorder=1)


def preparar(ax, titulo):
    ax.set_facecolor(FONDO); ax.set_aspect("equal"); ax.invert_yaxis(); ax.axis("off")
    ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")


def zoom(ax, p, idx, margen):
    x0, y0 = p[idx].min(0) - margen
    x1, y1 = p[idx].max(0) + margen
    ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)


def main():
    p = puntos_de_un_rostro()
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 4), facecolor=FONDO, gridspec_kw={"width_ratios": [1, 1.1, 1.1]})

    preparar(a, "a) Malla de 468 puntos")
    malla(a, p, C.FACE_LANDMARKS_TESSELATION, "#e6e5e0", 0.4)
    malla(a, p, C.FACE_LANDMARKS_CONTOURS, GRIS, 0.8)
    a.scatter(*p.T, s=1, color=GRIS, zorder=2)
    a.scatter(*p[OJO_DER + OJO_IZQ].T, s=14, color=AZUL, zorder=3, label="EAR")
    a.scatter(*p[BOCA].T, s=14, color=NARANJA, zorder=3, label="MAR")
    a.legend(loc="lower center", ncol=2, fontsize=8, frameon=False, labelcolor=TINTA, bbox_to_anchor=(0.5, -0.08))

    preparar(b, "b) Ojo derecho (EAR)")
    malla(b, p, C.FACE_LANDMARKS_RIGHT_EYE, GRIS, 1)
    p1, p2, p3, p4, p5, p6 = OJO_DER
    for i, j, estilo in [(p2, p6, "-"), (p3, p5, "-"), (p1, p4, "--")]:
        b.plot(*p[[i, j]].T, color=AZUL, linewidth=1.4, linestyle=estilo, zorder=2)
    for n, i in enumerate(OJO_DER, 1):
        b.scatter(*p[i], s=40, color=AZUL, zorder=3, edgecolor=FONDO, linewidth=1)
        dy = 6 if n in (2, 3) else -6 if n in (5, 6) else 0
        dx = -6 if n == 1 else 6 if n == 4 else 0
        b.annotate(f"p{n} ({i})", p[i], xytext=(dx * 1.6, dy * 1.6), textcoords="offset points", fontsize=8, color=TINTA,
                   ha="right" if n == 1 else "left" if n == 4 else "center",
                   va="bottom" if n in (2, 3) else "top" if n in (5, 6) else "center")
    zoom(b, p, OJO_DER, 14)

    preparar(c, "c) Boca (MAR)")
    malla(c, p, C.FACE_LANDMARKS_LIPS, GRIS, 1)
    for i, j in BOCA_VERT:
        c.plot(*p[[i, j]].T, color=NARANJA, linewidth=1.4, zorder=2)
    c.plot(*p[list(BOCA_HORIZ)].T, color=NARANJA, linewidth=1.4, linestyle="--", zorder=2)
    for i in BOCA:
        c.scatter(*p[i], s=40, color=NARANJA, zorder=3, edgecolor=FONDO, linewidth=1)
    for i, j in BOCA_VERT:
        c.annotate(str(i), p[i], xytext=(0, 6), textcoords="offset points", fontsize=8, color=TINTA, ha="center", va="bottom")
        c.annotate(str(j), p[j], xytext=(0, -6), textcoords="offset points", fontsize=8, color=TINTA, ha="center", va="top")
    for i, dx, ha in [(BOCA_HORIZ[0], -6, "right"), (BOCA_HORIZ[1], 6, "left")]:
        c.annotate(str(i), p[i], xytext=(dx, 0), textcoords="offset points", fontsize=8, color=TINTA, ha=ha, va="center")
    zoom(c, p, BOCA, 16)

    fig.text(0.5, 0.01, "Líneas continuas: distancias verticales (numerador). Línea punteada: distancia horizontal (denominador).",
             ha="center", fontsize=8, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SALIDA / "fig_puntos_ear_mar.png", dpi=300)
    print("Guardada:", SALIDA / "fig_puntos_ear_mar.png")


if __name__ == "__main__":
    main()
