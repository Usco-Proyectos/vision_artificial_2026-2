"""M2: bostezo con MobileNetV3-Small sobre el recorte de la boca de YawDD (etiquetas YawDD+).

- Misma red, entrada, aumento de datos e hiperparámetros que M1 (ver entrenar_m1.py y la Tabla 4).
- Diferencia: el bostezo es solo el 7 % de los fotogramas. En vez de usar el entrenamiento
  submuestreado (1 096 + 1 096 imágenes), se usan los 15 619 fotogramas de entrenamiento con pesos
  de clase ω_c = N / (2·N_c) en la entropía cruzada (ec. 12). Así no se botan los fotogramas de
  "hablando", que son justo los que más se parecen a un bostezo.
- Validación y prueba conservan la proporción real (7 %), así que la exactitud engaña: lo que
  importa es el F1, la sensibilidad y la precisión promedio (AP).

Uso (desde la carpeta src):
    python -m modelos.entrenar_m2 --rapido     # prueba corta (1 época, pocas imágenes)
    python -m modelos.entrenar_m2              # entrenamiento completo (reanudable)
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn import functional as F

import config
from modelos.entrenar_m1 import (EPOCAS, UMBRAL, cargadores, crear_red, entrenar, evaluar,
                                 medir_latencia, metricas)


class EntropiaConPesos(nn.Module):
    """Entropía cruzada binaria con un peso por clase (ec. 12)."""

    def __init__(self, w0, w1):
        super().__init__()
        self.w0, self.w1 = w0, w1

    def forward(self, z, y):
        pesos = torch.where(y == 1, self.w1, self.w0)
        return (F.binary_cross_entropy_with_logits(z, y, reduction="none") * pesos).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hilos", type=int, default=2, help="procesos que leen imágenes")
    ap.add_argument("--rapido", action="store_true", help="prueba corta: pocas imágenes y 1 época")
    args = ap.parse_args()

    torch.manual_seed(config.SEMILLA); np.random.seed(config.SEMILLA)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre = "m2" + ("_rapido" if args.rapido else "")

    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    d = d[d.fuente == "yawdd"]
    ent, val, pru = (d[d.particion == p] for p in ("entrenamiento", "validacion", "prueba"))
    if args.rapido:  # estratificado para que alcancen a salir bostezos
        ent, val, pru = (pd.concat(g.sample(min(len(g), 256), random_state=0) for _, g in x.groupby("bostezo"))
                         for x in (ent, val, pru))

    n, n1 = len(ent), int(ent.bostezo.sum())
    w0, w1 = n / (2 * (n - n1)), n / (2 * n1)
    print(f"{nombre} en {disp}: entrenamiento {n} ({n1} bostezos), validación {len(val)}, prueba {len(pru)}")
    print(f"pesos de clase: ω0 = {w0:.3f} (no bostezo), ω1 = {w1:.3f} (bostezo)")

    c_ent, c_val, c_pru = cargadores(ent, val, pru, "ruta_boca", lambda x: x["bostezo"], args.hilos)
    red = crear_red().to(disp)
    perdida = EntropiaConPesos(w0, w1)
    mejor_f1 = entrenar(red, c_ent, c_val, perdida, nombre, 1 if args.rapido else EPOCAS, disp)

    # Prueba final con el mejor modelo (según F1 de validación)
    _, y_pru, p_pru = evaluar(red, c_pru, disp, perdida)
    resultado = {"modelo": nombre, "prueba": metricas(y_pru, p_pru),
                 "precision_promedio_AP": float(average_precision_score(y_pru, p_pru)),
                 "proporcion_bostezo_prueba": float(y_pru.mean()),
                 "pesos_clase": {"w0": w0, "w1": w1}, **medir_latencia(red, disp),
                 "dispositivo": torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu",
                 "mejor_f1_validacion": mejor_f1}
    # Desglose para la discusión: ¿confunde hablar con bostezar? ¿le afectan las gafas?
    pru = pru.assign(y=y_pru, p=p_pru, pred=(p_pru >= UMBRAL).astype(int))
    for cond in ["situacion", "gafas", "genero"]:
        resultado[f"por_{cond}"] = {
            str(k): {"n": len(g), "bostezos": int(g.y.sum()),
                     "exactitud": round(float((g.pred == g.y).mean()), 4),
                     "marcados_como_bostezo": round(float(g.pred.mean()), 4)}
            for k, g in pru.groupby(cond)}
    (config.REPORTES / f"{nombre}_resultado.json").write_text(json.dumps(resultado, indent=2), encoding="utf8")
    pru[["id", "sujeto", "video", "situacion", "y", "p"]].to_csv(
        config.REPORTES / f"{nombre}_predicciones_prueba.csv", index=False)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
