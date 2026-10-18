"""Prueba en vivo de M1 (ojo cerrado) con la cámara web o con fotos.

Cada fotograma pasa por el mismo procesamiento del entrenamiento (MediaPipe, enderezado,
recorte del ojo de 96x96 en gris) y M1 da la probabilidad de ojo cerrado de cada ojo.
En pantalla se ven: la probabilidad de cada ojo, el promedio, el EAR (para comparar),
los dos recortes que ve la red y el % de fotogramas con ojo cerrado en los últimos 3 s
(lo que será z1 para M5).

Uso (desde la carpeta src):
    python -m modelos.probar_m1                       # busca sola la cámara que da imagen
    python -m modelos.probar_m1 --camara 1            # otra cámara
    python -m modelos.probar_m1 --imagen foto1.jpg foto2.jpg
Teclas: q o Esc para salir.
"""
import argparse
import time
from collections import deque

import cv2
import numpy as np
import torch
from torchvision import transforms

import config
from dataset.rostro import DetectorRostro
from modelos.entrenar_m1 import DESV, MEDIA, UMBRAL, a_tres_canales, crear_red

VENTANA_S = 3.3  # una ventana del sistema: 10 fotogramas a 3 FPS
VERDE, ROJO, BLANCO, GRIS = (80, 200, 80), (60, 60, 230), (255, 255, 255), (170, 170, 170)


class M1:
    def __init__(self):
        self.disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.red = crear_red().to(self.disp)
        self.red.load_state_dict(torch.load(config.RAIZ_DATOS / "modelos" / "m1_mejor.pt",
                                            map_location=self.disp, weights_only=True))
        self.red.eval()
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(a_tres_canales),
                                      transforms.Normalize(MEDIA, DESV)])

    @torch.no_grad()
    def __call__(self, *ojos):
        """Probabilidad de ojo cerrado para cada recorte gris de 96x96."""
        x = torch.stack([self.tf(o) for o in ojos]).to(self.disp)
        return torch.sigmoid(self.red(x).squeeze(1)).cpu().numpy()


def texto(img, t, pos, color=BLANCO, escala=0.6):
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, t, pos, cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def dibujar(img, r, p_der, p_izq, historial=None, fps=None):
    """Escribe los resultados sobre el fotograma (los acentos no se ven en OpenCV, por eso no van)."""
    if r is None:
        texto(img, "No se detecta rostro", (10, 30), ROJO, 0.8)
    else:
        p = (p_der + p_izq) / 2
        cerrado = p >= UMBRAL
        texto(img, "OJO CERRADO" if cerrado else "ojo abierto", (10, 35), ROJO if cerrado else VERDE, 1.0)
        texto(img, f"P(cerrado) promedio: {p:.2f}   (umbral {UMBRAL})", (10, 65))
        texto(img, f"ojo der: {p_der:.2f}   ojo izq: {p_izq:.2f}", (10, 90))
        texto(img, f"EAR: {(r.ear_der + r.ear_izq) / 2:.3f}  (cerrado si < 0.20)", (10, 115), GRIS)
        # Los recortes que ve la red, ampliados, en la esquina superior derecha
        for k, ojo in enumerate((r.ojo_der, r.ojo_izq)):
            grande = cv2.cvtColor(cv2.resize(ojo, (120, 120)), cv2.COLOR_GRAY2BGR)
            x = img.shape[1] - 130 * (2 - k)
            if x >= 0 and img.shape[0] >= 140:
                img[10:130, x:x + 120] = grande
                texto(img, "der" if k == 0 else "izq", (x + 4, 145), GRIS, 0.5)
    if historial:
        frac = np.mean([c for _, c in historial])
        texto(img, f"% ojo cerrado (ultimos {VENTANA_S:.0f} s): {frac:.0%}", (10, img.shape[0] - 40))
    if fps:
        texto(img, f"{fps:.0f} FPS", (10, img.shape[0] - 15), GRIS, 0.5)


def abrir_camara(indice):
    """Abre la cámara pedida o, si no se pide, la primera que da imagen real.
    Las cámaras virtuales (por ejemplo EShare) dan negro o una imagen fija: se saltan."""
    for i in ([indice] if indice is not None else range(4)):
        cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DirectShow abre más rápido en Windows
        if not cam.isOpened():
            continue
        cuadros = [f for ok, f in (cam.read() for _ in range(15)) if ok]
        real = len(cuadros) > 1 and cuadros[-1].mean() > 5 and cv2.absdiff(cuadros[1], cuadros[-1]).mean() > 0.1
        if real or indice is not None:
            print(f"Usando la cámara {i}")
            return cam
        print(f"Cámara {i}: negra o imagen fija (¿cámara virtual?), se salta")
        cam.release()
    raise SystemExit("No se encontró una cámara que dé imagen. Prueba con --camara 0, 1, 2...")


def con_camara(m1, indice):
    cam = abrir_camara(indice)
    detector = DetectorRostro(modo_video=True)
    historial = deque()  # (tiempo, cerrado) de los últimos VENTANA_S segundos
    t0, ultimo = time.time(), time.time()
    fps = None
    try:
        while True:
            ok, img = cam.read()
            if not ok:
                break
            img = cv2.flip(img, 1)  # como un espejo, más natural para probar
            ahora = time.time()
            r = detector.procesar(img, (ahora - t0) * 1000)
            p_der = p_izq = None
            if r is not None:
                p_der, p_izq = m1(r.ojo_der, r.ojo_izq)
                historial.append((ahora, (p_der + p_izq) / 2 >= UMBRAL))
            while historial and ahora - historial[0][0] > VENTANA_S:
                historial.popleft()
            dt = ahora - ultimo
            ultimo = ahora
            fps = 1 / dt if fps is None else 0.9 * fps + 0.1 / max(dt, 1e-6)
            dibujar(img, r, p_der, p_izq, historial, fps)
            cv2.imshow("Prueba de M1 - q para salir", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.release()
        detector.cerrar()
        cv2.destroyAllWindows()


def con_imagenes(m1, rutas):
    detector = DetectorRostro()
    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is None:
            print(f"{ruta}: no se pudo leer")
            continue
        r = detector.procesar(img)
        if r is None:
            print(f"{ruta}: no se detecta rostro")
            dibujar(img, None, None, None)
        else:
            p_der, p_izq = m1(r.ojo_der, r.ojo_izq)
            p = (p_der + p_izq) / 2
            print(f"{ruta}: der={p_der:.3f} izq={p_izq:.3f} promedio={p:.3f} -> "
                  f"{'CERRADO' if p >= UMBRAL else 'abierto'} (EAR {(r.ear_der + r.ear_izq) / 2:.3f})")
            dibujar(img, r, p_der, p_izq)
        cv2.imshow("Prueba de M1 - cualquier tecla para seguir", img)
        if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
            break
    detector.cerrar()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", type=int, help="número de cámara; sin esto se busca sola")
    ap.add_argument("--imagen", nargs="+", help="una o varias fotos en lugar de la cámara")
    args = ap.parse_args()
    m1 = M1()
    print(f"M1 cargado en {m1.disp}")
    con_imagenes(m1, args.imagen) if args.imagen else con_camara(m1, args.camara)


if __name__ == "__main__":
    main()
