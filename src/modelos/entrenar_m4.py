"""M4: somnolencia con SVM o Random Forest sobre los descriptores por ventana de UTA-RLDD (ecs. 7-10, 20-23).

- Entrada: x_w = [EAR medio, PERCLOS, parpadeos, MAR máximo], calculados igual que en la
  Figura 4 (analisis/figuras_uta.py). Mismas ventanas que M3: n_w >= 5 y solo KSS 0 y KSS 10.
- Se prueban SVM con kernel RBF (x_w estandarizado, z4 con sigmoide de Platt) y Random Forest
  (z4 = fracción de árboles, vía predict_proba) con una rejilla de hiperparámetros. Gana el
  que tenga mejor F1 en validación, igual que en las redes.
- Entrena con las ventanas balanceadas de entrenamiento, como M3.
- Guarda el modelo y la salida z4 de las ventanas del combinador y de prueba (para M5).
  Tarda menos de un minuto.

Uso (desde la carpeta src):
    python -m modelos.entrenar_m4                          # partición por sujeto (la real)
    python -m modelos.entrenar_m4 --particion aleatoria    # solo para medir la fuga
"""
import argparse
import json
import time
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import config
from analisis.figuras_uta import MIN_FOTOGRAMAS, descriptores_ventana
from modelos.entrenar_m1 import DIR_MODELOS, UMBRAL, metricas

X = ["ear_medio", "perclos", "parpadeos", "mar_max"]


def candidatos():
    """Rejilla de hiperparámetros: (nombre, modelo sin entrenar)."""
    for c, g in product([0.1, 1, 10, 100], [0.01, 0.1, 1, "scale"]):
        yield (f"svm C={c} gamma={g}",
               make_pipeline(StandardScaler(), SVC(C=c, gamma=g, probability=True, random_state=config.SEMILLA)))
    for n, prof, hoja in product([300], [3, 5, None], [1, 5, 20]):
        yield (f"rf arboles={n} profundidad={prof} hoja={hoja}",
               RandomForestClassifier(n, max_depth=prof, min_samples_leaf=hoja, random_state=config.SEMILLA, n_jobs=-1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", choices=["sujeto", "aleatoria"], default="sujeto")
    args = ap.parse_args()
    nombre = "m4" if args.particion == "sujeto" else "m4_aleatoria"
    col = "particion" if args.particion == "sujeto" else "particion_aleatoria"

    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    d = d[(d.fuente == "uta") & d.somnolencia.notna()]  # fuera KSS 5
    v = d.groupby("ventana").apply(descriptores_ventana, include_groups=False)
    v = v.join(d.groupby("ventana")[[col, "balanceado", "somnolencia"]].first())
    v = v[v.n >= MIN_FOTOGRAMAS]
    partes = {p: v[v[col] == p] for p in ("entrenamiento", "validacion", "combinador", "prueba")}
    if args.particion == "sujeto":  # en la aleatoria no hay balanceo; KSS 0 y 10 ya están parejos
        partes["entrenamiento"] = partes["entrenamiento"][partes["entrenamiento"].balanceado]
    print(f"{nombre}: " + ", ".join(f"{p} {len(x)} ventanas" for p, x in partes.items()))

    ent, val = partes["entrenamiento"], partes["validacion"]
    rejilla, mejor, mejor_f1 = [], None, -1.0
    for etiqueta, modelo in candidatos():
        modelo.fit(ent[X], ent.somnolencia)
        m = metricas(val.somnolencia.to_numpy(), modelo.predict_proba(val[X])[:, 1])
        rejilla.append({"modelo": etiqueta, "f1_val": m["f1"], "exactitud_val": m["exactitud"]})
        print(f"{etiqueta:40s} | f1_val {m['f1']:.4f} | exactitud_val {m['exactitud']:.4f}")
        if m["f1"] > mejor_f1:
            mejor, mejor_f1, mejor_etiqueta = modelo, m["f1"], etiqueta
    print(f"elegido: {mejor_etiqueta}")

    # Prueba final con el elegido
    pru = partes["prueba"]
    y, p = pru.somnolencia.to_numpy(), mejor.predict_proba(pru[X])[:, 1]
    if hasattr(mejor, "n_jobs"):
        mejor.set_params(n_jobs=1)  # con una ventana a la vez, repartir en varios núcleos cuesta más (61 ms vs. pocos ms)
    t0 = time.perf_counter()
    for _ in range(200):  # latencia de una sola ventana, como en la app
        mejor.predict_proba(pru[X].iloc[:1])
    resultado = {"modelo": nombre, "elegido": mejor_etiqueta, "prueba": metricas(y, p),
                 "precision_promedio_AP": float(average_precision_score(y, p)),
                 "latencia_ms_por_ventana": (time.perf_counter() - t0) / 200 * 1000,
                 "mejor_f1_validacion": mejor_f1,
                 "exactitud_por_sujeto": pd.Series((p >= UMBRAL) == (y == 1)).groupby(pru.sujeto.to_numpy())
                 .mean().round(4).to_dict(),
                 "rejilla": rejilla}
    if hasattr(mejor, "feature_importances_"):  # en el RF: cuánto pesa cada descriptor
        resultado["importancia"] = dict(zip(X, mejor.feature_importances_.round(4).tolist()))
    joblib.dump(mejor, DIR_MODELOS / f"{nombre}.joblib")
    (config.REPORTES / f"{nombre}_resultado.json").write_text(json.dumps(resultado, indent=2), encoding="utf8")

    # Salida z4 por ventana para entrenar y probar M5 (mismo formato que M3)
    pd.concat(pd.DataFrame({"ventana": x.index, "particion": nombre_p, "y": x.somnolencia.to_numpy(),
                            "z4": mejor.predict_proba(x[X])[:, 1]})
              for nombre_p, x in ((q, partes[q]) for q in ("combinador", "prueba"))
              ).to_csv(config.REPORTES / f"{nombre}_predicciones.csv", index=False)
    print(json.dumps({k: v for k, v in resultado.items() if k != "rejilla"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
