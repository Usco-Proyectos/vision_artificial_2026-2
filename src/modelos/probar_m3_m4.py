"""Prueba en vivo de M3 (LSTM sobre señales) y M4 (Random Forest sobre descriptores) con la cámara web.

Hace lo mismo que hará la app: toma un fotograma cada 1/3 s (3 FPS), arma la ventana con los
últimos 10 (3,3 s) y, si al menos 5 tienen rostro, calcula:
- M3: la secuencia s_t = [M1 ojo der, M1 ojo izq, M2 boca, EAR der, EAR izq, MAR] entra a la LSTM.
- M4: los descriptores de la ventana (EAR medio, PERCLOS, parpadeos, MAR máximo) entran al RF.
Las dos probabilidades se actualizan con cada fotograma nuevo (ventana deslizante).
Ojo: en prueba los dos aciertan ~60 % por ventana, así que lo que se ve es una tendencia.

Uso (desde la carpeta src):
    python -m modelos.probar_m3_m4                    # busca sola la cámara que da imagen
    python -m modelos.probar_m3_m4 --camara 1         # otra cámara
Teclas: q o Esc para salir.
"""
import argparse
import time
from collections import deque

import cv2
import joblib
import numpy as np
import pandas as pd
import torch

import config
from analisis.figuras_uta import MIN_FOTOGRAMAS, descriptores_ventana
from dataset.rostro import DetectorRostro
from modelos.entrenar_m1 import UMBRAL
from modelos.entrenar_m3 import T, CnnLstm
from modelos.entrenar_m4 import X
from modelos.probar_m1 import GRIS, M1, ROJO, VERDE, abrir_camara, texto
from modelos.probar_m2 import M2


class M3M4:
    def __init__(self):
        self.m1, self.m2 = M1(), M2()
        self.disp = self.m1.disp
        self.m3 = CnnLstm(senales=True).to(self.disp)  # la media y la desviación vienen dentro del .pt
        self.m3.load_state_dict(torch.load(config.RAIZ_DATOS / "modelos" / "m3_senales_mejor.pt",
                                           map_location=self.disp, weights_only=True))
        self.m3.eval()
        self.m4 = joblib.load(config.RAIZ_DATOS / "modelos" / "m4.joblib")

    def senales(self, r):
        """s_t de un fotograma con rostro (ec. 18)."""
        p_der, p_izq = self.m1(r.ojo_der, r.ojo_izq)
        return [float(p_der), float(p_izq), self.m2(r.boca), r.ear_der, r.ear_izq, r.mar]

    @torch.no_grad()
    def __call__(self, ventana):
        """ventana: lista de s_t (solo fotogramas con rostro, en orden). Devuelve (z3, z4)."""
        s = torch.tensor(ventana, dtype=torch.float32)
        x = torch.cat([s, torch.full((T - len(s), s.shape[1]), float("nan"))])[None].to(self.disp)
        z3 = torch.sigmoid(self.m3(x)).item()
        # mismos descriptores que en el entrenamiento; kss y sujeto no se usan, solo los pide la función
        d = pd.DataFrame(ventana, columns=["p_der", "p_izq", "p_boca", "ear_der", "ear_izq", "mar"])
        x4 = descriptores_ventana(d.assign(fotograma=range(len(d)), kss=0, sujeto=""))
        z4 = self.m4.predict_proba(pd.DataFrame([x4[X]]))[0, 1]
        return z3, z4


def dibujar(img, r, n_w, z3, z4):
    """Los acentos no se ven en OpenCV, por eso no van."""
    if r is None:
        texto(img, "No se detecta rostro", (10, 30), ROJO, 0.8)
    texto(img, f"ventana: {n_w}/{T} fotogramas con rostro (minimo {MIN_FOTOGRAMAS})", (10, img.shape[0] - 15), GRIS, 0.5)
    for k, (nombre, z) in enumerate((("M3 (LSTM)", z3), ("M4 (RF)", z4))):
        y = 65 + 35 * k
        if z is None:
            texto(img, f"{nombre}: esperando ventana...", (10, y), GRIS)
            continue
        texto(img, f"{nombre}: P(somnolencia) {z:.2f}", (10, y), ROJO if z >= UMBRAL else VERDE, 0.8)
        cv2.rectangle(img, (330, y - 15), (330 + int(200 * z), y - 2), ROJO if z >= UMBRAL else VERDE, -1)
        cv2.rectangle(img, (330, y - 15), (530, y - 2), GRIS, 1)


def con_camara(modelos, indice):
    cam = abrir_camara(indice)
    detector = DetectorRostro(modo_video=True)
    ventana = deque(maxlen=T)  # los últimos 10 fotogramas muestreados: s_t o None si no hubo rostro
    t0 = proximo = time.time()
    r, z3, z4 = None, None, None
    try:
        while True:
            ok, img = cam.read()
            if not ok:
                break
            img = cv2.flip(img, 1)  # como un espejo, más natural para probar
            ahora = time.time()
            if ahora >= proximo:  # muestreo a 3 FPS, igual que en el entrenamiento
                proximo = ahora + 1 / config.FPS_OBJETIVO
                r = detector.procesar(img, (ahora - t0) * 1000)
                ventana.append(None if r is None else modelos.senales(r))
                reales = [s for s in ventana if s is not None]
                if len(ventana) == T and len(reales) >= MIN_FOTOGRAMAS:
                    z3, z4 = modelos(reales)
                elif len(reales) < MIN_FOTOGRAMAS:
                    z3 = z4 = None
            dibujar(img, r, sum(s is not None for s in ventana), z3, z4)
            cv2.imshow("Prueba de M3 y M4 - q para salir", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.release()
        detector.cerrar()
        cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", type=int, help="número de cámara; sin esto se busca sola")
    args = ap.parse_args()
    modelos = M3M4()
    print(f"M1, M2, M3 y M4 cargados en {modelos.disp}")
    con_camara(modelos, args.camara)


if __name__ == "__main__":
    main()
