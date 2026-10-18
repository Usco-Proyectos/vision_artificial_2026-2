"""UTA-RLDD (videos) -> clips de 10 fotogramas a 3 FPS cada 20 s + tabla uta.csv.

Se procesa por partes: se descarga un pliegue en C:\\datasets\\crudo\\uta, se corre
este script y, si se pasa --borrar-videos, cada video se borra apenas se procesa.
Es reanudable: los videos que ya están en uta.csv se saltan.

Supuestos sobre la descarga (ver docs/fuentes/diseno/datasets-acceso.md):
- la etiqueta es el nombre del archivo: 0 (alerta), 5 (vigilancia baja), 10 (somnoliento);
  el sujeto 32 tiene el 10 partido en 10_1 y 10_2;
- el sujeto es la carpeta que contiene el video;
- el pliegue sale de una carpeta llamada "Fold<n>..." si existe.

Uso:  python -m dataset.procesar_uta [--borrar-videos]   (desde la carpeta src)
"""
import argparse
import hashlib
import re

import cv2
import pandas as pd
from tqdm import tqdm

import config
from dataset.rostro import DetectorRostro

EXTENSIONES = {".mov", ".mp4", ".m4v", ".avi"}
SALIDA = config.PROCESADO / "uta.csv"
NOMBRE_VIDEO = re.compile(r"(0|5|10)(_\d+)?")


def pliegue_de(ruta):
    for parte in ruta.parts:
        m = re.match(r"fold\s*(\d)", parte, re.I)
        if m:
            return int(m.group(1))
    return None


def procesar_video(video, detector):
    # Casi siempre el nombre es 0, 5 o 10; el sujeto 32 tiene el 10 partido en 10_1 y 10_2.
    parte = video.stem
    kss = int(parte.split("_")[0])
    sujeto_orig = video.parent.name
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    paso = max(round(fps / config.FPS_OBJETIVO), 1)
    entre_clips = int(round(fps * config.UTA_SEGUNDOS_ENTRE_CLIPS))
    largo_clip = paso * config.UTA_FOTOGRAMAS_CLIP

    filas, muestreados = [], 0
    for n_clip, inicio in enumerate(range(0, max(total - largo_clip, 1), entre_clips)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, inicio)
        for k in range(largo_clip):
            ok, frame = cap.read()
            if not ok:
                break
            if k % paso:
                continue
            idx = inicio + k
            muestreados += 1
            r = detector.procesar(frame)
            if r is None:
                continue
            base = config.RECORTES / "uta" / sujeto_orig / parte
            base.mkdir(parents=True, exist_ok=True)
            rutas = {}
            for nombre, img in (("ojo_der", r.ojo_der), ("ojo_izq", r.ojo_izq),
                                ("boca", r.boca), ("rostro", r.rostro)):
                destino = base / f"{idx:06d}_{nombre}.png"
                cv2.imwrite(str(destino), img)
                rutas[f"ruta_{nombre}"] = destino.relative_to(config.PROCESADO).as_posix()
            filas.append({
                "id": f"uta_{sujeto_orig}_{parte}_{idx:06d}",
                "fuente": "uta",
                "sujeto": f"uta_{sujeto_orig}",
                "pliegue": pliegue_de(video),
                "video": f"{sujeto_orig}/{video.name}",
                "ventana": f"uta_{sujeto_orig}_{parte}_c{n_clip:03d}",
                "fotograma": idx,
                **rutas,
                "ear_der": r.ear_der, "ear_izq": r.ear_izq, "mar": r.mar,
                "kss": kss,
                "somnolencia": {0: 0, 10: 1}.get(kss),  # el nivel 5 queda vacío
                "hash": hashlib.md5(r.rostro.tobytes()).hexdigest(),
            })
    cap.release()
    return filas, muestreados, fps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--borrar-videos", action="store_true",
                    help="borra cada video después de procesarlo (para ahorrar disco)")
    args = ap.parse_args()

    videos = sorted(v for v in config.UTA_DIR.rglob("*")
                    if v.suffix.lower() in EXTENSIONES and NOMBRE_VIDEO.fullmatch(v.stem))
    previo = pd.read_csv(SALIDA) if SALIDA.exists() else pd.DataFrame()
    hechos = set(previo["video"]) if len(previo) else set()
    pendientes = [v for v in videos if f"{v.parent.name}/{v.name}" not in hechos]
    print(f"UTA: {len(videos)} videos encontrados, {len(pendientes)} por procesar")

    detector = DetectorRostro()
    rep_path = config.REPORTES / "uta_retencion.csv"
    config.REPORTES.mkdir(parents=True, exist_ok=True)
    for video in tqdm(pendientes, desc="UTA"):
        filas, muestreados, fps = procesar_video(video, detector)
        # Se guarda después de cada video para poder reanudar si se corta.
        pd.DataFrame(filas).to_csv(SALIDA, mode="a", header=not SALIDA.exists(), index=False)
        pd.DataFrame([{"video": f"{video.parent.name}/{video.name}", "fps": fps,
                       "muestreados": muestreados, "con_rostro": len(filas)}]).to_csv(
            rep_path, mode="a", header=not rep_path.exists(), index=False)
        if args.borrar_videos and filas:
            video.unlink()
    detector.cerrar()

    df = pd.read_csv(SALIDA)
    print(f"UTA acumulado: {len(df)} registros, {df['sujeto'].nunique()} sujetos, "
          f"{df['ventana'].nunique()} ventanas")
    print(df["kss"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
