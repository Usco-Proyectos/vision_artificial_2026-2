"""
eval_fotos.py — v2: evaluación offline con imágenes estáticas + control de ausencia de sujeto.

Convención de nombres en la carpeta:
    despierto_XX.jpg     → ground truth 0 (persona despierta)
    somnoliento_XX.jpg   → ground truth 1 (persona somnolienta)
    nopersona_XX.jpg     → CONTROL: no hay sujeto (objeto, pared, etc.)

Uso:
    python eval_fotos.py data/fotos_prueba
"""
import time
import sys
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from app import config as C
from app.models_loader import cargar_modelos
from app.pipeline import Pipeline

C.MODO_PRUEBA_LOCAL = False   # ⭐ no contaminar el log de la prueba en vivo

REPETICIONES = 120             # ~6 s de video estático a 10 FPS


def fmt(x):
    return " None" if x is None else f"{x:.3f}"


def evaluar(pipe, frame, reps=REPETICIONES):
    pipe.reiniciar()
    res = None
    for _ in range(reps):
        res = pipe.procesar(frame.copy())
        if res["modelos"]["m3"]["estado"] == "ok":
            break                    
        time.sleep(0.1)               
    return res

def categoria(nombre: str):
    n = nombre.lower()
    if n.startswith("somnoliento"):
        return "persona", 1
    if n.startswith("nopersona"):
        return "control", None
    return "persona", 0


def evaluar(pipe, frame, reps=REPETICIONES):
    pipe.reiniciar()          # estado temporal limpio entre imágenes
    res = None
    for _ in range(reps):
        res = pipe.procesar(frame.copy())
    return res


def main():
    carpeta = Path(sys.argv[1] if len(sys.argv) > 1 else "data/fotos")
    rutas = sorted(p for p in carpeta.glob("*")
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not rutas:
        raise SystemExit(f"No hay imágenes en {carpeta}")

    print("Cargando modelos...")
    pipe = Pipeline(cargar_modelos())
    pipe.calentar()

    filas = []
    for ruta in rutas:
        cat, gt = categoria(ruta.stem)
        frame = cv2.imread(str(ruta))
        if frame is None:
            print(f"  [skip] no se pudo leer {ruta.name}")
            continue

        res = evaluar(pipe, frame)
        m = res["modelos"]
        rostro = bool(res["rostro_detectado"])
        pred = res["veredicto"]

        if cat == "control":
            ok = (not rostro) and pred == 0
            detalle = f"rostro={rostro} vered={pred}"
        else:
            ok = pred == gt
            detalle = f"gt={gt} pred={pred}"

        filas.append({
            "archivo": ruta.name, "categoria": cat, "gt": gt, "pred": pred,
            "rostro_detectado": rostro, "regla": res["regla_activada"],
            "m1_der": m["m1"]["prob_der"], "m1_izq": m["m1"]["prob_izq"],
            "m2_prob": m["m2"]["prob"], "m3_prob": m["m3"]["prob"],
            "m4_pred": m["m4"]["prediccion"],
            "latencia_ms": res["rendimiento"]["latencia_ms"],
        })
        print(f"[{'OK ' if ok else 'FAIL'}] {ruta.name:<26} {detalle:<24} "
              f"regla={res['regla_activada']:<24} "
              f"m1=({fmt(m['m1']['prob_der'])},{fmt(m['m1']['prob_izq'])}) m2={fmt(m['m2']['prob'])}")

    df = pd.DataFrame(filas)
    df.to_csv("resultados_fotos.csv", index=False)

    # ── Métricas de somnolencia (solo imágenes con persona) ──
    dp = df[df.categoria == "persona"]
    print("\n" + "=" * 60)
    print("MÉTRICAS DE SOMNOLENCIA (imágenes con persona)")
    print("=" * 60)
    if len(dp) and dp["gt"].nunique() == 2:
        print(f"Accuracy : {accuracy_score(dp['gt'], dp['pred']):.4f}")
        print(f"Precisión: {precision_score(dp['gt'], dp['pred'], zero_division=0):.4f}")
        print(f"Recall   : {recall_score(dp['gt'], dp['pred'], zero_division=0):.4f}")
        print(f"F1       : {f1_score(dp['gt'], dp['pred'], zero_division=0):.4f}")
    else:
        print(f"Aciertos: {(dp['gt'] == dp['pred']).sum()}/{len(dp)}")

    # ── Control de ausencia de sujeto ──
    dc = df[df.categoria == "control"]
    print("\n" + "=" * 60)
    print("CONTROL DE AUSENCIA DE SUJETO (negative control)")
    print("=" * 60)
    if len(dc):
        sin_rostro = (~dc["rostro_detectado"]).sum()
        sin_alerta = (dc["pred"] == 0).sum()
        print(f"Imágenes sin sujeto:       {len(dc)}")
        print(f"Sin rostro detectado:      {sin_rostro}/{len(dc)}")
        print(f"Sin alertas (veredicto 0): {sin_alerta}/{len(dc)}")
        if sin_rostro < len(dc):
            print("  MediaPipe reportó rostro en al menos una imagen sin sujeto:")
            print(dc[dc.rostro_detectado][["archivo", "regla"]].to_string(index=False))
    else:
        print("(no hay imágenes de control en la carpeta)")

    print("\n Tabla guardada en resultados_fotos.csv")


if __name__ == "__main__":
    main()