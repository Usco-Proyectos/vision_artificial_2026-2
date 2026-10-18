"""Rutas y constantes del proyecto (ver docs/fuentes/diseno/dataset.md).

Los datos viven fuera de OneDrive, en C:\\datasets. Se puede cambiar con la
variable de entorno DATOS_SOMNOLENCIA.
"""
import os
from pathlib import Path

# En el repositorio de GitHub los datos van en <repo>/datos; aquí, en C:\datasets.
_DATOS_REPO = Path(__file__).resolve().parent.parent / "datos"
RAIZ_DATOS = Path(os.environ.get("DATOS_SOMNOLENCIA", _DATOS_REPO if _DATOS_REPO.is_dir() else r"C:\datasets"))

CRUDO = RAIZ_DATOS / "crudo"
PROCESADO = RAIZ_DATOS / "procesado"
RECORTES = PROCESADO / "recortes"
REPORTES = PROCESADO / "reportes"
MODELO_LANDMARKS = RAIZ_DATOS / "modelos" / "face_landmarker.task"

# Fuentes crudas
MRL_DIR = CRUDO / "mrl" / "mrlEyes_2018_01"
YAWDD_VIDEOS = CRUDO / "yawdd"                      # aquí se descomprime YawDD.rar.gz
# Etiquetas YawDD+ (clon del repositorio de la otra sesión; no se copian: son ~287 000 archivos)
YAWDD_PLUS_LABELS = Path(r"C:\datos-somnolencia\crudo\yawdd_plus\labels")
UTA_DIR = CRUDO / "uta"

# Muestreo (sección 3 del diseño)
FPS_OBJETIVO = 3            # tasa dentro de una ventana, igual que en la app
UTA_FOTOGRAMAS_CLIP = 10    # 10 fotogramas a 3 FPS = ~3,3 s por ventana
UTA_SEGUNDOS_ENTRE_CLIPS = 20

# Tamaño de los recortes guardados (escala de grises)
TAM_OJO = 96
TAM_BOCA = 96
TAM_ROSTRO = 160

SEMILLA = 42
