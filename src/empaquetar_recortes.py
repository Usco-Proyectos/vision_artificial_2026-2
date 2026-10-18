"""Comprime los recortes en un .zip por fuente, para subirlos como archivos del Release de GitHub.

Uso (desde C:\\proyecto-somnolencia):
  C:\\datasets\\.venv\\Scripts\\python src\\empaquetar_recortes.py

Lee C:\\datasets\\procesado\\recortes y escribe release\\recortes_<fuente>.zip.
Para usarlos después: descomprimir cada zip dentro de datos\\procesado (quedan en datos\\procesado\\recortes\\...).
Si un zip ya existe, se salta; para rehacerlo, se borra a mano.
"""
import sys
import zipfile
from pathlib import Path

ORIGEN = Path(r"C:\datasets\procesado")
DESTINO = Path(__file__).resolve().parent.parent / "release"
LIMITE_GITHUB = 2 * 1024**3  # tamaño máximo de un archivo de Release


def main():
    DESTINO.mkdir(exist_ok=True)
    for carpeta in sorted((ORIGEN / "recortes").iterdir()):
        zip_ruta = DESTINO / f"recortes_{carpeta.name}.zip"
        if zip_ruta.exists():
            print(f"{zip_ruta.name} ya existe, se salta")
            continue
        temporal = zip_ruta.with_suffix(".zip.parcial")  # si se corta, no queda un zip a medias con el nombre final
        n = 0
        # ZIP_STORED: los PNG ya vienen comprimidos, comprimir otra vez solo gasta tiempo
        with zipfile.ZipFile(temporal, "w", zipfile.ZIP_STORED) as z:
            for archivo in carpeta.rglob("*"):
                if archivo.is_file():
                    z.write(archivo, archivo.relative_to(ORIGEN).as_posix())
                    n += 1
                    if n % 20000 == 0:
                        print(f"  {carpeta.name}: {n} archivos")
        temporal.rename(zip_ruta)
        tam = zip_ruta.stat().st_size
        print(f"{zip_ruta.name}: {n} archivos, {tam / 2**20:.0f} MB")
        if tam >= LIMITE_GITHUB:
            sys.exit(f"{zip_ruta.name} pasa de 2 GB: GitHub no lo acepta, hay que partirlo")


if __name__ == "__main__":
    main()
