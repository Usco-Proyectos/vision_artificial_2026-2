"""MRL Eye Dataset -> recortes de ojo 96x96 + tabla mrl.csv.

Las imágenes ya vienen recortadas al ojo, así que no se usa MediaPipe.
Nombre: sXXXX_imgId_genero_gafas_ojo_reflejo_luz_sensor.png
(ojo: 0 cerrado, 1 abierto; ver docs/fuentes/diseno/datasets-acceso.md).

Uso:  python -m dataset.procesar_mrl   (desde la carpeta src)
"""
import hashlib

import cv2
import pandas as pd
from tqdm import tqdm

import config

CAMPOS = ["sujeto", "img", "genero", "gafas", "ojo", "reflejo", "luz", "sensor"]


def main():
    salida = config.RECORTES / "mrl"
    archivos = sorted(config.MRL_DIR.rglob("*.png"))
    if not archivos:
        raise SystemExit(f"No hay imágenes en {config.MRL_DIR}")

    filas, descartes = [], []
    for ruta in tqdm(archivos, desc="MRL"):
        partes = ruta.stem.split("_")
        if len(partes) != len(CAMPOS):
            descartes.append((str(ruta), "nombre con formato inesperado"))
            continue
        meta = dict(zip(CAMPOS, partes))
        destino = salida / meta["sujeto"] / f"{ruta.stem}.png"
        # Reanudable: si el recorte ya existe (de una corrida cortada), se reutiliza.
        img = cv2.imread(str(destino), cv2.IMREAD_GRAYSCALE) if destino.exists() else None
        if img is None:
            img = cv2.imread(str(ruta), cv2.IMREAD_GRAYSCALE)
            if img is None:
                descartes.append((str(ruta), "imagen dañada"))
                continue
            img = cv2.resize(img, (config.TAM_OJO, config.TAM_OJO), interpolation=cv2.INTER_AREA)
            destino.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(destino), img)
        filas.append({
            "id": f"mrl_{ruta.stem}",
            "fuente": "mrl",
            "sujeto": f"mrl_{meta['sujeto']}",
            "ruta_ojo": destino.relative_to(config.PROCESADO).as_posix(),
            "ojo": int(meta["ojo"]),
            "gafas": int(meta["gafas"]),
            "genero": "mujer" if meta["genero"] == "1" else "hombre",
            "iluminacion": "buena" if meta["luz"] == "1" else "mala",
            "reflejo": int(meta["reflejo"]),
            "sensor": meta["sensor"],
            "hash": hashlib.md5(img.tobytes()).hexdigest(),
        })

    config.REPORTES.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(filas)
    df.to_csv(config.PROCESADO / "mrl.csv", index=False)
    pd.DataFrame(descartes, columns=["archivo", "motivo"]).to_csv(config.REPORTES / "mrl_descartes.csv", index=False)
    print(f"MRL: {len(df)} registros, {df['sujeto'].nunique()} sujetos, {len(descartes)} descartados")
    print(df["ojo"].value_counts().rename({0: "cerrado", 1: "abierto"}).to_string())


if __name__ == "__main__":
    main()
