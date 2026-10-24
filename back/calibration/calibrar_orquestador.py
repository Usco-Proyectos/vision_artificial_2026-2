"""
calibrar_orquestador.py — Valida y calibra la cascada de seguridad SIN imágenes ni modelos reales.
CORREGIDO: Filtra solo filas con ground truth (somnolencia) y usa proxies EAR/MAR para M1/M2.
"""
import argparse
import sys
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config as C
from app.orchestrator import CascadaSeguridad, Umbrales

RUTA_CSV = "../datos/procesado/dataset.csv"
EXACTITUD_M1 = 0.9657
EXACTITUD_M2 = 0.9826
EXACTITUD_M3 = 0.85
RHO_RUIDO = 0.0
FPS_CSV = 30

# Umbrales para proxies de M1 y M2 basados en EAR/MAR
UMBRAL_EAR_CERRADO = 0.25  # Si EAR promedio < 0.25, ojo cerrado
UMBRAL_MAR_BOSTEZO = 0.50  # Si MAR > 0.50, bostezo

KSS_SOMNOLENCIA = 7
EAR_UMBRAL_M4 = 0.25
MAR_UMBRAL_M4 = 0.50

N_REPETICIONES = 30
N_REPETICIONES_GRID = 10
SEMILLA = 42
GRID_PROB_M1 = [0.3, 0.4, 0.5, 0.6, 0.7]
GRID_FRAMES_OJOS = [5, 10, 15, 20, 30]
# ════════════════════════════════════════════════════

STRIDE = max(1, round(FPS_CSV / C.FPS_M3))
ESCALA = FPS_CSV / C.FPS_ENTRADA


@dataclass
class Corrida:
    y: np.ndarray
    m1: list
    m2: np.ndarray
    m3: list
    m4: list


def cargar_videos(ruta: Path) -> list:
    """Carga solo filas con ground truth (somnolencia) y agrupa por video/sujeto."""
    df = pd.read_csv(ruta)
    
    # FILTRAR: Solo filas con ground truth
    df = df[df['somnolencia'].notna()].copy()
    
    # ⭐ FIX: Resetear índices para que sean consecutivos desde 0
    df = df.reset_index(drop=True)
    
    if len(df) == 0:
        sys.exit("Error: No hay filas con 'somnolencia' en el CSV")
    
    print(f"Filas con ground truth: {len(df)}")
    
    # Convertir columnas numéricas
    for c in ("ear_der", "ear_izq", "mar", "kss"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            df[c] = df[c].fillna(df[c].median())
    
    # Ground truth
    y = df["somnolencia"].astype(int).values
    
    # EAR promedio de ambos ojos
    ear_prom = (df["ear_der"] + df["ear_izq"]) / 2
    
    # Proxy para M1: ojo cerrado si EAR < umbral
    ojo_cerrado_proxy = (ear_prom < UMBRAL_EAR_CERRADO).astype(int).values
    
    # Proxy para M2: bostezo si MAR > umbral
    bostezo_proxy = (df["mar"] > UMBRAL_MAR_BOSTEZO).astype(int).values
    
    # M3: basado en KSS o somnolencia
    kss = df["kss"] if "kss" in df.columns else pd.Series(np.nan, index=df.index)
    m3_verdad = np.where(kss.notna(), kss >= KSS_SOMNOLENCIA, y == 1).astype(int)
    
    # Agrupar por video o sujeto
    if "video" in df.columns:
        grupo = df["video"].fillna("global").astype(str)
    elif "sujeto" in df.columns:
        grupo = df["sujeto"].astype(str)
    else:
        grupo = pd.Series(["global"] * len(df))
    
    videos = []
    for _, g in df.groupby(grupo, sort=False):
        idx = g.index  # Ahora estos índices son consecutivos
        videos.append({
            "y": y[idx],
            "ojo": ojo_cerrado_proxy[idx],
            "bos": bostezo_proxy[idx],
            "m3": m3_verdad[idx],
            "ear": ear_prom.values[idx],
            "mar": df["mar"].values[idx]
        })
    
    return videos

def sigma(exactitud: float) -> float:
    assert 0.5 < exactitud < 1, "la exactitud debe estar en (0.5, 1)"
    return 0.25 / NormalDist().inv_cdf(exactitud)


def ruido(n: int, s: float, rng) -> np.ndarray:
    e = rng.normal(0, 1, n)
    if RHO_RUIDO > 0:
        k = np.sqrt(1 - RHO_RUIDO ** 2)
        for i in range(1, n):
            e[i] = RHO_RUIDO * e[i - 1] + k * e[i]
    return s * e


def simular_video(v: dict, rng) -> Corrida:
    n = len(v["y"])
    s1, s2, s3 = sigma(EXACTITUD_M1), sigma(EXACTITUD_M2), sigma(EXACTITUD_M3)
    
    # M1: probabilidades con ruido basado en proxy de EAR
    base_prob = np.where(v["ojo"] == 1, 0.75, 0.25)
    noise_der = ruido(n, s1, rng)
    noise_izq = ruido(n, s1, rng)
    m1_der = np.clip(base_prob + noise_der, 0.01, 0.99)
    m1_izq = np.clip(base_prob + noise_izq, 0.01, 0.99)
    m1 = [[float(d), float(i)] for d, i in zip(m1_der, m1_izq)]
    
    # M2: probabilidades con ruido basado en proxy de MAR
    m2 = np.clip(np.where(v["bos"] == 1, 0.75, 0.25) + ruido(n, s2, rng), 0.01, 0.99)
    
    # M3 y M4: muestreo cada STRIDE frames
    m3, m4 = [None] * n, [None] * n
    lbl, ear, mar = (deque(maxlen=C.VENTANA_M3) for _ in range(3))
    
    for t in range(0, n, STRIDE):
        lbl.append(v["m3"][t])
        ear.append(v["ear"][t])
        mar.append(v["mar"][t])
        
        if len(lbl) == C.VENTANA_M3:
            centro = 0.75 if np.mean(lbl) >= 0.5 else 0.25
            m3[t] = int(np.clip(centro + rng.normal(0, s3), 0.01, 0.99) >= C.UMBRAL_PROB_M3)
            m4[t] = int(np.mean(ear) < EAR_UMBRAL_M4 or np.mean(mar) > MAR_UMBRAL_M4)
    
    return Corrida(v["y"], m1, m2, m3, m4)


def ejecutar(corridas, umbrales: Umbrales):
    ys, ps, reglas = [], [], Counter()
    for c in corridas:
        casc = CascadaSeguridad(umbrales)
        for t in range(len(c.y)):
            r = casc.step(c.m1[t], c.m2[t], c.m3[t], c.m4[t])
            ps.append(r.veredicto)
            reglas[r.regla] += 1
        ys.append(c.y)
    return np.concatenate(ys), np.array(ps), reglas


def metricas(y, p) -> dict:
    cm = confusion_matrix(y, p, labels=[0, 1])
    tn, fp = cm[0]
    return {
        "acc": accuracy_score(y, p),
        "recall": recall_score(y, p, zero_division=0),
        "prec": precision_score(y, p, zero_division=0),
        "f1": f1_score(y, p, zero_division=0),
        "fpr": fp / max(1, fp + tn),
        "cm": cm
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=RUTA_CSV)
    args = ap.parse_args()
    
    if not args.csv.exists():
        sys.exit(f"Error: No se encontró el CSV en {args.csv}")
    
    videos = cargar_videos(args.csv)
    n_frames = sum(len(v["y"]) for v in videos)
    umbral_ojos = max(1, round(C.UMBRAL_FRAMES_OJOS * ESCALA))
    base = Umbrales(frames_ojos=umbral_ojos)
    
    print(f"Videos/flujos: {len(videos)} | FPS_CSV={FPS_CSV} | stride M3={STRIDE} | "
          f"umbral ojos={umbral_ojos} frames | RHO={RHO_RUIDO}")
    print(f"Prevalencia somnolencia (GT): {np.mean(np.concatenate([v['y'] for v in videos])):.3f}\n")
    
    # Evaluación con umbrales base
    rng = np.random.default_rng(SEMILLA)
    res, cms, reglas_tot = [], np.zeros((2, 2)), Counter()
    
    for _ in range(N_REPETICIONES):
        y, p, reglas = ejecutar([simular_video(v, rng) for v in videos], base)
        m = metricas(y, p)
        res.append(m)
        cms += m["cm"]
        reglas_tot += reglas
    
    f = lambda k: f"{np.mean([r[k] for r in res]):.4f} ± {np.std([r[k] for r in res]):.4f}"
    print(f"=== Cascada con umbrales base ({N_REPETICIONES} repeticiones) ===")
    print(f"Accuracy : {f('acc')}\nRecall (somnoliento): {f('recall')}\nPrecisión: {f('prec')}\n"
          f"F1       : {f('f1')}\nFPR (falsas alarmas): {f('fpr')}")
    print("\nMatriz de confusión media:")
    print(pd.DataFrame((cms / N_REPETICIONES).round(1), index=["real 0", "real 1"],
                       columns=["pred 0", "pred 1"]))
    tot = sum(reglas_tot.values())
    print("\nRegla que decide (%):")
    for k, c in reglas_tot.most_common():
        print(f"  {k:<36s}{100 * c / tot:6.2f}%")
    
    # Barrido de hiperparámetros
    print(f"\n=== Barrido UMBRAL_PROB_M1 × UMBRAL_FRAMES_OJOS ===")
    filas = []
    sims = [[simular_video(v, np.random.default_rng(SEMILLA + 1000 + i)) for v in videos]
            for i in range(N_REPETICIONES_GRID)]
    for pm1 in GRID_PROB_M1:
        fila = {}
        for fr in GRID_FRAMES_OJOS:
            ms = [metricas(*ejecutar(s, Umbrales(prob_m1=pm1, frames_ojos=max(1, round(fr * ESCALA))))[:2]) for s in sims]
            fila[f"{fr} fr"] = f"{np.mean([m['recall'] for m in ms]):.3f}/{np.mean([m['fpr'] for m in ms]):.3f}"
        filas.append(fila)
    print(pd.DataFrame(filas, index=[f"M1≥{p}" for p in GRID_PROB_M1]).to_string())


if __name__ == "__main__":
    main()