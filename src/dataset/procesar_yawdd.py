"""YawDD (videos) + YawDD+ (etiquetas por fotograma) -> recortes + tabla yawdd.csv.

Etiquetas YawDD+: <labels>/Mirror/<video>/labels/<fotograma 8 dígitos>_<x>.txt, en formato
YOLO: "clase cx cy ancho alto". La etiqueta es la CLASE de adentro del archivo
(1 = bostezo, 0 = no bostezo), no el sufijo del nombre (en esta copia siempre es _0).
El fotograma empieza en 0.

Se toma 1 de cada 10 fotogramas (30 FPS -> 3 FPS).

Sujeto: número + género del nombre del video ("1-MaleSunGlasses-Yawning" -> "1-Male").
Así, si la misma persona aparece con y sin gafas, queda como un solo sujeto
(agrupar de más no produce fuga; agrupar de menos sí).
Solo se usa la cámara Mirror: la numeración de Dash es distinta y no se puede
saber si son las mismas personas, así que mezclarla podría producir fuga.

Es reanudable: guarda después de cada video y salta los que ya están en el reporte.

Uso:  python -m dataset.procesar_yawdd   (desde la carpeta src)
"""
import hashlib
import re

import cv2
import pandas as pd
from tqdm import tqdm

import config
from dataset.rostro import DetectorRostro

PASO = 10
SALIDA = config.PROCESADO / "yawdd.csv"
REPORTE = config.REPORTES / "yawdd_retencion.csv"
GAFAS = {"NoGlasses": 0, "Glasses": 1, "SunGlasses": 2}
# Algunos nombres traen extras: "19-MaleGlassesmoustache-Normal", "23-MaleGlassesBeard-Talking&Yawning"
PATRON = re.compile(r"^(\d+)-(Male|Female)(NoGlasses|Glasses|SunGlasses)[A-Za-z]*-([\w&]+)$")


def etiquetas_video(carpeta):
    """{fotograma: etiqueta} a partir de los nombres de archivo de YawDD+."""
    etiquetas = {}
    for f in (carpeta / "labels").glob("*.txt"):
        num = int(f.stem.split("_")[0])
        if num % PASO:  # solo se leen los fotogramas que se van a muestrear
            continue
        contenido = f.read_text().split()
        if contenido:
            etiquetas[num] = int(contenido[0])
    return etiquetas


def guardar(filas, resumen):
    """Agrega las filas de un video y su línea de reporte (el reporte va después:
    si se corta en medio, el video se repite y no queda a medias)."""
    if filas:
        pd.DataFrame(filas).to_csv(SALIDA, mode="a", header=not SALIDA.exists(), index=False)
    pd.DataFrame([resumen]).to_csv(REPORTE, mode="a", header=not REPORTE.exists(), index=False)


def main():
    videos = {v.stem: v for v in config.YAWDD_VIDEOS.rglob("*.avi")}
    carpetas = sorted((config.YAWDD_PLUS_LABELS / "Mirror").iterdir())
    if not videos:
        raise SystemExit(f"No hay videos .avi en {config.YAWDD_VIDEOS}: falta descomprimir YawDD")

    config.REPORTES.mkdir(parents=True, exist_ok=True)
    hechos = set(pd.read_csv(REPORTE)["video"]) if REPORTE.exists() else set()
    pendientes = [c for c in carpetas if c.name not in hechos]
    print(f"YawDD: {len(carpetas)} videos con etiquetas, {len(pendientes)} por procesar")

    detector = DetectorRostro()
    for carpeta in tqdm(pendientes, desc="YawDD"):
        filas = []
        m = PATRON.match(carpeta.name)
        video = videos.get(carpeta.name)
        if not m or video is None:
            guardar([], {"video": carpeta.name, "estado": "sin video o nombre inesperado"})
            continue
        num, genero, gafas, situacion = m.groups()
        situacion = situacion.replace("&yawning", "&Yawning")
        sujeto = f"yawdd_{num}-{genero}"
        etiquetas = etiquetas_video(carpeta)
        cap = cv2.VideoCapture(str(video))
        idx, muestreados, sin_rostro = 0, 0, 0
        while True:
            if idx % PASO:
                if not cap.grab():
                    break
                idx += 1
                continue
            ok, frame = cap.read()
            if not ok:
                break
            if idx in etiquetas:
                muestreados += 1
                r = detector.procesar(frame)
                if r is None:
                    sin_rostro += 1
                else:
                    base = config.RECORTES / "yawdd" / carpeta.name
                    base.mkdir(parents=True, exist_ok=True)
                    rutas = {}
                    for nombre, img in (("ojo_der", r.ojo_der), ("ojo_izq", r.ojo_izq),
                                        ("boca", r.boca), ("rostro", r.rostro)):
                        destino = base / f"{idx:08d}_{nombre}.png"
                        cv2.imwrite(str(destino), img)
                        rutas[f"ruta_{nombre}"] = destino.relative_to(config.PROCESADO).as_posix()
                    filas.append({
                        "id": f"yawdd_{carpeta.name}_{idx:08d}",
                        "fuente": "yawdd",
                        "sujeto": sujeto,
                        "video": carpeta.name,
                        "fotograma": idx,
                        **rutas,
                        "ear_der": r.ear_der, "ear_izq": r.ear_izq, "mar": r.mar,
                        "bostezo": etiquetas[idx],
                        "gafas": GAFAS[gafas],
                        "genero": "mujer" if genero == "Female" else "hombre",
                        "situacion": situacion,
                        "hash": hashlib.md5(r.boca.tobytes()).hexdigest(),
                    })
            idx += 1
        cap.release()
        guardar(filas, {"video": carpeta.name, "estado": "ok", "muestreados": muestreados,
                        "sin_rostro": sin_rostro})
    detector.cerrar()

    df = pd.read_csv(SALIDA)
    rep = pd.read_csv(REPORTE)
    total = rep.get("muestreados", pd.Series(dtype=float)).sum()
    print(f"YawDD: {len(df)} registros de {df['sujeto'].nunique()} sujetos; "
          f"retención {len(df) / max(total, 1):.1%}; videos con problemas: {(rep['estado'] != 'ok').sum()}")
    print(df["bostezo"].value_counts().rename({0: "no bostezo", 1: "bostezo"}).to_string())


if __name__ == "__main__":
    main()
