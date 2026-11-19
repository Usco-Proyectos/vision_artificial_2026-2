"""
analizar_prueba_real.py
Calcula automáticamente las métricas de tu prueba en vivo (celular + PC).
"""
import pandas as pd
import numpy as np
from sklearn.metrics import (confusion_matrix, precision_score,
                             recall_score, f1_score, accuracy_score)
import matplotlib.pyplot as plt


# ════════════════════════════════════════════════════════════
INTERVALOS_SOMNOLIENTOS = [
    (180, 210),   # Min 3:00 - 3:30 → ojos cerrados
    (360, 390),   # Min 6:00 - 6:30 → bostezos sostenidos
    (480, 540),   # Min 8:00 - 9:00 → micro-sueño
]
# ════════════════════════════════════════════════════════════

RUTA_CSV = "../data/log_prueba_real.csv"

# 1. Cargar el log que generó el backend durante la prueba
df = pd.read_csv(RUTA_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Segundos transcurridos desde el primer frame
t0 = df["timestamp"].iloc[0]
df["seg"] = (df["timestamp"] - t0).dt.total_seconds()

# 2. Métricas de rendimiento del sistema (100% automáticas)
fps_promedio = df["fps_backend"].mean()
latencia_promedio = df["latencia_ms"].mean()
latencia_p95 = np.percentile(df["latencia_ms"], 95)

print("=" * 55)
print("RENDIMIENTO DEL SISTEMA (END-TO-END)")
print("=" * 55)
print(f"Frames procesados:     {len(df)}")
print(f"Duración de la prueba: {df['seg'].iloc[-1]:.1f} segundos")
print(f"FPS promedio:          {fps_promedio:.2f} fps")
print(f"Latencia promedio:     {latencia_promedio:.2f} ms")
print(f"Latencia percentil 95: {latencia_p95:.2f} ms")

# 3. Construir la verdad terrena a partir de TUS intervalos
def verdad(seg):
    for a, b in INTERVALOS_SOMNOLIENTOS:
        if a <= seg <= b:
            return 1
    return 0

df["real"] = df["seg"].apply(verdad)
df["pred"] = df["veredicto"]

# 4. Métricas de detección (automáticas)
y_true = df["real"].tolist()
y_pred = df["pred"].tolist()

tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

print("\n" + "=" * 55)
print("DETECCIÓN EN VIVO (sistema vs. tu bitácora)")
print("=" * 55)
print(f"Verdaderos Negativos (TN): {tn}  → despierto y NO alertó")
print(f"Falsos Positivos     (FP): {fp}  → despierto pero SÍ alertó")
print(f"Falsos Negativos     (FN): {fn}  → somnoliento y NO alertó")
print(f"Verdaderos Positivos (TP): {tp}  → somnoliento y SÍ alertó")
print("-" * 55)
print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
print(f"Precisión: {precision_score(y_true, y_pred, zero_division=0):.4f}")
print(f"Recall   : {recall_score(y_true, y_pred, zero_division=0):.4f}")
print(f"F1-Score : {f1_score(y_true, y_pred, zero_division=0):.4f}")

# 5. Figura 1: Matriz de confusión para el artículo
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Pred: Despierto", "Pred: Somnoliento"])
plt.yticks([0, 1], ["Real: Despierto", "Real: Somnoliento"])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]}", ha="center", va="center")
plt.title("Matriz de Confusión - Prueba en Tiempo Real")
plt.ylabel("Valor Real")
plt.xlabel("Predicción del Sistema")
plt.tight_layout()
plt.savefig("matriz_confusion_prueba_real.png", dpi=300)
print("\n Guardado: matriz_confusion_prueba_real.png")

# 6. Figura 2: Probabilidades en el tiempo (zonas rojas = tus simulaciones)
plt.figure(figsize=(12, 5))
plt.plot(df["seg"], df["m1_prob_der"], label="M1 ojo der", alpha=0.6)
plt.plot(df["seg"], df["m3_prob"], label="M3 somnolencia", linewidth=2)
for a, b in INTERVALOS_SOMNOLIENTOS:
    plt.axvspan(a, b, color="red", alpha=0.15)
plt.axhline(0.5, color="r", linestyle="--", label="Umbral 0.5")
plt.xlabel("Segundos desde el inicio de la prueba")
plt.ylabel("Probabilidad")
plt.title("Probabilidades del sistema vs. eventos simulados (zonas rojas)")
plt.legend()
plt.tight_layout()
plt.savefig("grafico_probabilidades_tiempo.png", dpi=300)
print("✅ Guardado: grafico_probabilidades_tiempo.png")