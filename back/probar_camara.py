"""Prueba del pipeline completo con la cámara del portátil, sin servidor ni celular.

Cada fotograma pasa por lo mismo que en /ws (MediaPipe, M1-M4 y la cascada) y se muestra
la regla en pantalla. Al salir (tecla q) imprime cuántos fotogramas dio cada regla y la latencia.
Guarda todo en latencias_prueba.csv para el artículo.

Uso (desde back/):  python probar_camara.py [--camara 0] [--fps 30] [--segundos 60]
"""
import argparse
import time
from collections import Counter

import cv2
import pandas as pd

from app.models_loader import cargar_modelos
from app.pipeline import Pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camara", type=int, default=0)
    ap.add_argument("--fps", type=float, default=30, help="fotogramas por segundo que se le mandan al pipeline")
    ap.add_argument("--segundos", type=float, default=0, help="se cierra solo después de este tiempo (0 = hasta la tecla q)")
    args = ap.parse_args()

    pipe = Pipeline(cargar_modelos())
    pipe.calentar()
    cam = cv2.VideoCapture(args.camara)
    if not cam.isOpened():
        raise SystemExit(f"No se pudo abrir la cámara {args.camara}")

    filas, reglas = [], Counter()
    inicio = time.perf_counter()
    try:
        while not args.segundos or time.perf_counter() - inicio < args.segundos:
            t = time.perf_counter()
            ok, frame = cam.read()
            if not ok:
                break
            r = pipe.procesar(frame)
            reglas[r["regla_activada"]] += 1
            m = r["modelos"]
            filas.append({"frame": r["frame"], "regla": r["regla_activada"], "rostro": r["rostro_detectado"],
                          "m3_prob": m["m3"]["prob"], "m4_prob": m["m4"]["prob"], "kss": m["m4"]["kss_estimado"],
                          "latencia_ms": r["rendimiento"]["latencia_ms"], "fps": r["rendimiento"]["fps_backend"],
                          **{f"ms_{k}": v for k, v in r["rendimiento"]["tiempos_ms"].items()}})

            color = (0, 0, 255) if r["veredicto"] else (0, 200, 0)
            cv2.putText(frame, r["regla_activada"], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            p4 = m["m4"]["prob"]
            cv2.putText(frame, f"M3 {m['m3']['prob'] or 0:.2f}  M4 {p4 or 0:.2f}  KSS {m['m4']['kss_estimado'] or 0:.1f}  {r['rendimiento']['latencia_ms']:.0f} ms",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("prueba cascada (q para salir)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(max(0.0, 1 / args.fps - (time.perf_counter() - t)))
    finally:
        cam.release()
        cv2.destroyAllWindows()
        pipe.cerrar()

    d = pd.DataFrame(filas)
    if d.empty:
        return
    d.to_csv("latencias_prueba.csv", index=False)
    print(f"\n{len(d)} fotogramas, rostro en {d.rostro.mean():.0%}")
    print("fotogramas por regla:", dict(reglas))
    print(d[["latencia_ms", "ms_extraccion", "ms_m1_m2", "ms_m3_m4"]].iloc[10:].describe().round(1).loc[["mean", "50%", "max"]])
    print(f"FPS del backend (mediana): {d.fps.iloc[10:].median():.1f}")
    print("Guardado: latencias_prueba.csv")


if __name__ == "__main__":
    main()
