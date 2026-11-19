import time
import numpy as np
import pandas as pd
import torch

from app import config as C
from app.inferencia import predecir_m1, predecir_m2, predecir_m3, predecir_m4
from app.models_loader import cargar_modelos
from app.vision import preparar


mod = cargar_modelos()

def bench(fn, args, n=100, warm=20):
    for _ in range(warm):
        fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn(*args)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1000)
    ts = np.array(ts)
    return f"media={ts.mean():.2f} ms, mediana={np.median(ts):.2f} ms, desv={ts.std():.2f} ms"

z_ojo = np.zeros((3, 96, 96), np.float32)
z_m3 = np.zeros((1, C.VENTANA_M3, C.DIM_SENALES_M3), np.float32)
df_m4 = pd.DataFrame([[0.3, 0.1, 2, 0.4]], columns=C.FEATURES_M4)

crop = np.zeros((180, 220, 3), np.uint8)   # recorte crudo simulado (BGR)
print("M1 prep+inferencia:", bench(predecir_m1, (mod.m1, [
    preparar(crop, C.TAM_M1, C.M1_ESCALA_GRISES),
    preparar(crop, C.TAM_M1, C.M1_ESCALA_GRISES)])))
print("M2 prep+inferencia:", bench(predecir_m2, (mod.m2,
    preparar(crop, C.TAM_M2, C.M2_ESCALA_GRISES))))

print("M3 (ventana):     ", bench(predecir_m3, (mod.m3, z_m3)))
print("M4 (descriptores):", bench(predecir_m4, (mod.m4, df_m4)))