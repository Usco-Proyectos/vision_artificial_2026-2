"""
config.py — Hiperparámetros FIJOS del orquestador y rutas.
VERSIÓN DEFINITIVA unificada. Sin dependencias pesadas (no importa torch).
"""
from pathlib import Path

# ───────── Rutas ─────────
BASE_DIR = Path(__file__).resolve().parent.parent
RUTA_M1 = BASE_DIR / "models" / "m1.pt"
RUTA_M2 = BASE_DIR / "models" / "m2.pt"
RUTA_M3 = BASE_DIR / "models" / "m3.pt"
RUTA_M4 = BASE_DIR / "models" / "m4.joblib"
RUTA_KSS = BASE_DIR / "models" / "kss.joblib"  # regresión lineal sobre los mismos 4 descriptores de M4
RUTA_CSV_CALIBRACION = BASE_DIR / "data" / "prueba.csv"

DEVICE_PREFERIDO = "cuda"     
NUM_HILOS_TORCH = 4

UMBRAL_PROB_M1 = 0.5
UMBRAL_PROB_M2 = 0.5
UMBRAL_PROB_M3 = 0.5
UMBRAL_PROB_M4 = 0.5
UMBRAL_FRAMES_OJOS = 3    
UMBRAL_FRAMES_BOSTEZO = 1     
VOTOS_MINIMOS = 2           # C2: alerta si al menos 2 de M1-M4 votan somnolencia (ver orchestrator.py)

FPS_ENTRADA = 30
FPS_M3 = 3                  # M3 y M4 toman una muestra cada 1/3 s, sin importar a qué FPS lleguen las fotos
VENTANA_M3 = 10             # 10 muestras a 3 FPS = ~3,3 s, igual que en el entrenamiento
MIN_FRAMES_VALIDOS_M3 = 5   # en el entrenamiento solo se usaron ventanas con >= 5 fotogramas con rostro

M3_MODO = "senales"
SENALES_M3 = ["p_ojo_der", "p_ojo_izq", "p_boca", "ear_der", "ear_izq", "mar"]
DIM_SENALES_M3 = len(SENALES_M3)   # = 6

TAM_M1 = (96, 96)
TAM_M2 = (96, 96)
M1_ESCALA_GRISES = True
M2_ESCALA_GRISES = True
MEDIA = (0.485, 0.456, 0.406)   
STD = (0.229, 0.224, 0.225)

FEATURES_M4 = ["ear_medio", "perclos", "parpadeos", "mar_max"]
UMBRAL_EAR_CERRADO = 0.20     


EAR_CAL_A, EAR_CAL_B = 1.0, 0.0
MAR_CAL_A, MAR_CAL_B = 1.0, 0.0

INTERCAMBIAR_OJOS = False     

MARGEN_OJO = 0.8
MARGEN_BOCA = 0.5
MAX_LADO_PROCESO = 480
MIN_FRAMES_M4 = 3  