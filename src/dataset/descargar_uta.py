"""Descarga UTA-RLDD desde Google Drive, una parte a la vez, en C:\\datasets\\crudo\\uta.

Son 10 zips (Fold1_part1 ... Fold5_part2), ~11 GB cada uno. La descarga es
reanudable: si se corta, se vuelve a correr el mismo comando.

Uso:  python -m dataset.descargar_uta Fold1_part1 [Fold1_part2 ...]   (desde la carpeta src)
      python -m dataset.descargar_uta --lista
"""
import argparse
import time
import zipfile

import requests
from tqdm import tqdm

import config

URL = "https://drive.usercontent.google.com/download"
PEDAZO = 64 * 1024 * 1024
REINTENTOS = 10  # ~1,5 h de espera en total antes de rendirse
PARTES = {
    "Fold1_part1": "16x8afSOwfbdGAjUbfRFvwGj1ie8Lf5Do",
    "Fold1_part2": "1qPoJ06jeznytP_BGJ25hxdQy7eYTTjyk",
    "Fold2_part1": "12uVMVLv_kRXzQb4QDdru9RK3G9hzfpPQ",
    "Fold2_part2": "1nYGUpqE3x5ZeRLgDeMlUN4K_ccPjM2pe",
    "Fold3_part1": "1-JhE5TfkNyr4UbQk6PSkkpI7z-SChkWS",
    "Fold3_part2": "1LZU5KfJkFMj2pIkGwb3RjHDazoUfxSYQ",
    "Fold4_part1": "1IYKdrjvawf90bG_cfeoWA4rZSymllFbD",
    "Fold4_part2": "1zJlASBwABBgVSULQLiia0R_Eqar8_B3D",
    "Fold5_part1": "16w30wEgcAbaUr4gx9o4sl_2TbIunD5yA",
    "Fold5_part2": "1loOQE1-hoNM5dIOjXHI2Ss5NjjoPZcX6",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("partes", nargs="*")
    p.add_argument("--lista", action="store_true")
    args = p.parse_args()

    if args.lista or not args.partes:
        print("\n".join(PARTES))
        return

    config.UTA_DIR.mkdir(parents=True, exist_ok=True)
    for parte in args.partes:
        destino = config.UTA_DIR / f"{parte}.zip"
        print(f"Descargando {parte} -> {destino}")
        descargar(PARTES[parte], destino)


def pedir(id_drive, inicio, fin):
    """Pide los bytes [inicio, fin]. Drive da 'cuota excedida' si el rango es
    abierto (bytes=0-), pero sí entrega rangos cerrados, así que se baja por pedazos."""
    r = requests.get(URL, params={"id": id_drive, "export": "download", "confirm": "t"},
                     headers={"Range": f"bytes={inicio}-{fin}"}, stream=True, timeout=60)
    tipo = r.headers.get("Content-Type", "")
    if r.status_code != 206 or "text/html" in tipo:
        raise requests.RequestException(f"Drive respondió {r.status_code} {tipo} (¿cuota excedida?)")
    return r


def descargar(id_drive, destino):
    ya = destino.stat().st_size if destino.exists() else 0
    total = int(pedir(id_drive, 0, 0).headers["Content-Range"].split("/")[1])
    with open(destino, "ab") as f, \
            tqdm(total=total, initial=ya, unit="B", unit_scale=True, mininterval=30) as barra:
        while ya < total:
            fin = min(ya + PEDAZO, total) - 1
            for intento in range(REINTENTOS):
                try:
                    r = pedir(id_drive, ya, fin)
                    datos = b"".join(r.iter_content(1 << 20))
                    if len(datos) == fin - ya + 1:
                        break
                except requests.RequestException as e:
                    espera = min(30 * 2 ** intento, 900)  # 30 s, 1 min, 2 min ... hasta 15 min
                    print(f"\n  {e}; reintento {intento + 1} en {espera} s", flush=True)
                    time.sleep(espera)
            else:
                raise SystemExit("  Drive sigue sin entregar; volver a correr más tarde "
                                 "(la descarga sigue donde quedó).")
            f.write(datos)
            f.flush()
            ya += len(datos)
            barra.update(len(datos))
    zipfile.ZipFile(destino).close()  # falla si el zip quedó corrupto
    print(f"  listo: {total / 1e9:.1f} GB, zip válido")


if __name__ == "__main__":
    main()
