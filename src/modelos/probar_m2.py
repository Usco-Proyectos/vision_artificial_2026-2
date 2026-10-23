"""Prueba en vivo de M2 (bostezo) con la cámara web o con fotos.

Cada fotograma pasa por el mismo procesamiento del entrenamiento (MediaPipe, enderezado,
recorte de la boca en gris) y M2 da la probabilidad de bostezo.
En pantalla se ven: la probabilidad, el MAR (para comparar), el recorte que ve la red y el
máximo de la probabilidad en los últimos 3 s (lo que será z2 para M5).

Uso (desde la carpeta src):
    python -m modelos.probar_m2                       # busca sola la cámara que da imagen
    python -m modelos.probar_m2 --camara 1            # otra cámara
    python -m modelos.probar_m2 --imagen foto1.jpg foto2.jpg
Teclas: q o Esc para salir.
"""
import argparse
import time
from collections import deque

import cv2
import torch
from torchvision import transforms

import config
from dataset.rostro import DetectorRostro
from modelos.entrenar_m1 import DESV, MEDIA, UMBRAL, a_tres_canales, crear_red
from modelos.probar_m1 import GRIS, ROJO, VENTANA_S, VERDE, abrir_camara, texto


class M2:
    def __init__(self):
        self.disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.red = crear_red().to(self.disp)
        self.red.load_state_dict(torch.load(config.RAIZ_DATOS / "modelos" / "m2_mejor.pt",
                                            map_location=self.disp, weights_only=True))
        self.red.eval()
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(a_tres_canales),
                                      transforms.Normalize(MEDIA, DESV)])

    @torch.no_grad()
    def __call__(self, boca):
        """Probabilidad de bostezo para el recorte gris de la boca."""
        return torch.sigmoid(self.red(self.tf(boca)[None].to(self.disp))).item()


def dibujar(img, r, p, historial=None, fps=None):
    """Escribe los resultados sobre el fotograma (los acentos no se ven en OpenCV, por eso no van)."""
    if r is None:
        texto(img, "No se detecta rostro", (10, 30), ROJO, 0.8)
    else:
        bostezo = p >= UMBRAL
        texto(img, "BOSTEZO" if bostezo else "sin bostezo", (10, 35), ROJO if bostezo else VERDE, 1.0)
        texto(img, f"P(bostezo): {p:.2f}   (umbral {UMBRAL})", (10, 65))
        texto(img, f"MAR: {r.mar:.3f}", (10, 90), GRIS)
        # El recorte que ve la red, ampliado, en la esquina superior derecha
        grande = cv2.cvtColor(cv2.resize(r.boca, (120, 120)), cv2.COLOR_GRAY2BGR)
        x = img.shape[1] - 130
        if x >= 0 and img.shape[0] >= 140:
            img[10:130, x:x + 120] = grande
            texto(img, "boca", (x + 4, 145), GRIS, 0.5)
    if historial:
        texto(img, f"max P(bostezo) (ultimos {VENTANA_S:.0f} s): {max(q for _, q in historial):.2f}",
              (10, img.shape[0] - 40))
    if fps:
        texto(img, f"{fps:.0f} FPS", (10, img.shape[0] - 15), GRIS, 0.5)


def con_camara(m2, indice):
    cam = abrir_camara(indice)
    detector = DetectorRostro(modo_video=True)
    historial = deque()  # (tiempo, probabilidad) de los últimos VENTANA_S segundos
    t0 = ultimo = time.time()
    fps = None
    try:
        while True:
            ok, img = cam.read()
            if not ok:
                break
            img = cv2.flip(img, 1)  # como un espejo, más natural para probar
            ahora = time.time()
            r = detector.procesar(img, (ahora - t0) * 1000)
            p = None
            if r is not None:
                p = m2(r.boca)
                historial.append((ahora, p))
            while historial and ahora - historial[0][0] > VENTANA_S:
                historial.popleft()
            dt, ultimo = ahora - ultimo, ahora
            fps = 1 / max(dt, 1e-6) if fps is None else 0.9 * fps + 0.1 / max(dt, 1e-6)
            dibujar(img, r, p, historial, fps)
            cv2.imshow("Prueba de M2 - q para salir", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.release()
        detector.cerrar()
        cv2.destroyAllWindows()


def con_imagenes(m2, rutas):
    detector = DetectorRostro()
    for ruta in rutas:
        img = cv2.imread(ruta)
        if img is None:
            print(f"{ruta}: no se pudo leer")
            continue
        r = detector.procesar(img)
        p = None if r is None else m2(r.boca)
        print(f"{ruta}: no se detecta rostro" if r is None else
              f"{ruta}: P(bostezo)={p:.3f} -> {'BOSTEZO' if p >= UMBRAL else 'sin bostezo'} (MAR {r.mar:.3f})")
        dibujar(img, r, p)
        cv2.imshow("Prueba de M2 - cualquier tecla para seguir", img)
        if cv2.waitKey(0) & 0xFF in (ord("q"), 27):
            break
    detector.cerrar()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--camara", type=int, help="número de cámara; sin esto se busca sola")
    ap.add_argument("--imagen", nargs="+", help="una o varias fotos en lugar de la cámara")
    args = ap.parse_args()
    m2 = M2()
    print(f"M2 cargado en {m2.disp}")
    con_imagenes(m2, args.imagen) if args.imagen else con_camara(m2, args.camara)


if __name__ == "__main__":
    main()
