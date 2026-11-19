"""
analizar_prueba_real.py — v2
Métricas de la prueba en vivo: rendimiento, desglose por etapa y detección.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

RUTA_CSV = "../data/log_prueba_real_v2.csv"

INTERVALOS_SOMNOLIENTOS = [
    (180, 210),   # ojos cerrados
    (360, 390),   # bostezos
    (480, 540),   # micro-sueño
]

df = pd.read_csv(RUTA_CSV)
df["timestamp"] = pd.to_datetime(df["timestamp"])
t0 = df["timestamp"].iloc[0]
df["seg"] = (df["timestamp"] - t0).dt.total_seconds()

# ═══════════ 1. RENDIMIENTO DEL SISTEMA ═══════════
d = df.iloc[1:]   # excluye el primer frame (warm-up)

print("=" * 60)
print("RENDIMIENTO DEL SISTEMA (servidor, sin red ni captura)")
print("=" * 60)
print(f"Frames totales:        {len(df)}")
print(f"Duración de prueba:    {df['seg'].iloc[-1]:.1f} s")
print(f"FPS promedio:          {d['fps_backend'].mean():.2f} fps")
print(f"Latencia media:        {d['latencia_ms'].mean():.2f} ms")
print(f"Latencia mediana:      {d['latencia_ms'].median():.2f} ms")
print(f"Latencia p95:          {np.percentile(d['latencia_ms'], 95):.2f} ms")

def stats(x, nombre):
    print(f"{nombre:<32} media={x.mean():6.2f} ms | mediana={x.median():6.2f} ms | "
          f"p95={np.percentile(x, 95):6.2f} ms | n={len(x)}")

print("\n--- Desglose por etapa ---")
stats(d["extraccion_ms"], "Extracción (MediaPipe):")
stats(d.loc[d["rostro"] == 1, "m1_m2_ms"], "M1+M2 (frames con rostro):")
stats(d.loc[d["muestreo_m3"] == 1, "m3_m4_ms"], "M3+M4 (frames muestreados):")

# ═══════════ 2. DETECCIÓN (vs. ground truth por intervalos) ═══════════
def verdad(seg):
    return 1 if any(a <= seg <= b for a, b in INTERVALOS_SOMNOLIENTOS) else 0

df["real"] = df["seg"].apply(verdad)
df["pred"] = df["veredicto"]

tn, fp, fn, tp = confusion_matrix(df["real"], df["pred"], labels=[0, 1]).ravel()

print("\n" + "=" * 60)
print("DETECCIÓN EN VIVO")
("=" * 60)
print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")
print(f"Accuracy : {accuracy_score(df['real'], df['pred']):.4f}")
print(f"Precisión: {precision_score(df['real'], df['pred'], zero_division=0):.4f}")
print(f"Recall   : {recall_score(df['real'], df['pred'], zero_division=0):.4f}")
print(f"F1-Score : {f1_score(df['real'], df['pred'], zero_division=0):.4f}")

print("\n--- Falsas alarmas (FP) por regla ---")
print(df.loc[(df.real == 0) & (df.pred == 1), "regla"].value_counts().to_string())
print("\n--- Reglas activadas (total) ---")
print(df["regla"].value_counts().to_string())

# ═══════════ 3. FIGURAS PARA EL ARTÍCULO ═══════════
# Figura 1: matriz de confusión
cm = confusion_matrix(df["real"], df["pred"], labels=[0, 1])
plt.figure(figsize=(6, 5))
plt.imshow(cm, interpolation="nearest", cmap="Blues")
plt.colorbar()
plt.xticks([0, 1], ["Pred: Despierto", "Pred: Somnoliento"])
plt.yticks([0, 1], ["Real: Despierto", "Real: Somnoliento"])
umbral = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i, j]:d}", ha="center", va="center",
                 color="white" if cm[i, j] > umbral else "black")
plt.title("Matriz de confusión - prueba en tiempo real")
plt.ylabel("Valor real"); plt.xlabel("Predicción del sistema")
plt.tight_layout(); plt.savefig("fig_matriz_confusion.png", dpi=300)

# Figura 2: probabilidades vs. eventos simulados
plt.figure(figsize=(12, 5))
plt.plot(df["seg"], df["m1_prob_der"], label="M1 ojo der", alpha=0.6)
plt.plot(df["seg"], df["m3_prob"], label="M3 somnolencia", linewidth=2)
for a, b in INTERVALOS_SOMNOLIENTOS:
    plt.axvspan(a, b, color="red", alpha=0.15)
plt.axhline(0.5, color="r", linestyle="--", label="Umbral 0.5")
plt.xlabel("Segundos desde inicio"); plt.ylabel("Probabilidad")
plt.title("Probabilidades vs. eventos simulados (zonas rojas)")
plt.legend(); plt.tight_layout(); plt.savefig("fig_probabilidades.png", dpi=300)

# Figura 3: latencia media por etapa
etapas = ["Extracción", "M1+M2", "M3+M4"]
medias = [d["extraccion_ms"].mean(),
          d.loc[d["rostro"] == 1, "m1_m2_ms"].mean(),
          d.loc[d["muestreo_m3"] == 1, "m3_m4_ms"].mean()]
plt.figure(figsize=(7, 4.5))
plt.bar(etapas, medias, color=["#3b82f6", "#10b981", "#f59e0b"])
plt.ylabel("Latencia media (ms)")
plt.title("Latencia media por etapa del pipeline")
plt.tight_layout(); plt.savefig("fig_latencia_etapas.png", dpi=300)

print("\n Figuras guardadas: fig_matriz_confusion.png, fig_probabilidades.png, fig_latencia_etapas.png")