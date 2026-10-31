"""Los 4 modelos de ML clásico del curso sobre los descriptores por ventana de UTA-RLDD.

Misma entrada y mismas ventanas que M4: x_w = [EAR medio, PERCLOS, parpadeos, MAR máximo],
n_w >= 5, partición por sujeto. Cada familia se ajusta en validación (mejor F1) y se evalúa
una sola vez en prueba:

  - Regresión lineal: predice el nivel KSS como número (0, 5 o 10). Se entrena con las
    ventanas de KSS 0, 5 y 10. Para compararla con los demás, "somnoliento" si predice >= 5.
    Además se reportan el error absoluto medio (MAE) y el R² sobre todas las ventanas de prueba.
  - KNN, SVM y Random Forest: clasifican alerta (KSS 0) contra somnoliento (KSS 10), con el
    entrenamiento balanceado, igual que M4. SVM y RF usan la misma rejilla de entrenar_m4.

Guarda cada modelo en DIR_MODELOS/clasico_<familia>.joblib y todo en reportes/clasicos_resultado.json.
Tarda un par de minutos (la mayor parte es calcular los descriptores de cada ventana).

Uso (desde la carpeta src):  python -m modelos.entrenar_clasicos
"""
import json
import time
from itertools import product

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from analisis.figuras_uta import MIN_FOTOGRAMAS, descriptores_ventana
from modelos.entrenar_m1 import DIR_MODELOS, UMBRAL, metricas
from modelos.entrenar_m4 import X, candidatos

KSS_CORTE = 5  # la regresión dice "somnoliento" si el KSS estimado es >= 5


def rejilla(familia):
    if familia == "knn":
        for k, peso in product([3, 5, 11, 21, 41, 81], ["uniform", "distance"]):
            yield f"knn k={k} pesos={peso}", make_pipeline(StandardScaler(), KNeighborsClassifier(k, weights=peso))
    else:  # svm y rf: la misma rejilla de M4
        yield from ((e, m) for e, m in candidatos() if e.startswith(familia))


def latencia_ms(modelo, x):
    if hasattr(modelo, "n_jobs"):
        modelo.set_params(n_jobs=1)  # una ventana a la vez, como en la app
    t0 = time.perf_counter()
    for _ in range(200):
        modelo.predict(x.iloc[:1])
    return (time.perf_counter() - t0) / 200 * 1000


def por_sujeto(acierto, sujetos):
    return pd.Series(acierto).groupby(sujetos).mean().round(4).to_dict()


def regresion(partes):
    """Regresión lineal sobre el KSS (0, 5, 10); no tiene hiperparámetros que ajustar."""
    ent = partes["entrenamiento"]
    modelo = make_pipeline(StandardScaler(), LinearRegression()).fit(ent[X], ent.kss)
    pru = partes["prueba"]
    kss_est = modelo.predict(pru[X])
    binaria = pru.kss.isin([0, 10]).to_numpy()  # para comparar con los clasificadores
    y = (pru.kss[binaria] == 10).to_numpy().astype(float)
    pred = (kss_est[binaria] >= KSS_CORTE).astype(float)
    lineal = modelo[-1]
    return modelo, {
        "elegido": "regresion lineal (sin hiperparametros)",
        "prueba": metricas(y, pred),
        "mae_kss": float(mean_absolute_error(pru.kss, kss_est)),
        "r2_kss": float(r2_score(pru.kss, kss_est)),
        "kss_estimado_medio_por_nivel": pd.Series(kss_est).groupby(pru.kss.to_numpy()).mean().round(2).to_dict(),
        "coeficientes_estandarizados": dict(zip(X, lineal.coef_.round(3).tolist())),
        "intercepto": round(float(lineal.intercept_), 3),
        "latencia_ms_por_ventana": latencia_ms(modelo, pru[X]),
        "exactitud_por_sujeto": por_sujeto(pred == y, pru.sujeto[binaria].to_numpy()),
    }


def clasificador(familia, partes):
    ent = partes["entrenamiento"]
    ent = ent[(ent.kss != 5) & ent.balanceado]
    val, pru = (partes[p][partes[p].kss != 5] for p in ("validacion", "prueba"))
    filas, mejor, mejor_f1 = [], None, -1.0
    for etiqueta, modelo in rejilla(familia):
        modelo.fit(ent[X], ent.somnolencia)
        m = metricas(val.somnolencia.to_numpy(), modelo.predict_proba(val[X])[:, 1])
        filas.append({"modelo": etiqueta, "f1_val": m["f1"], "exactitud_val": m["exactitud"]})
        if m["f1"] > mejor_f1:
            mejor, mejor_f1, elegido = modelo, m["f1"], etiqueta
    y, p = pru.somnolencia.to_numpy(), mejor.predict_proba(pru[X])[:, 1]
    print(f"  {familia}: elegido {elegido} (F1 validación {mejor_f1:.3f})")
    return mejor, {
        "elegido": elegido, "mejor_f1_validacion": mejor_f1,
        "prueba": metricas(y, p), "precision_promedio_AP": float(average_precision_score(y, p)),
        "latencia_ms_por_ventana": latencia_ms(mejor, pru[X]),
        "exactitud_por_sujeto": por_sujeto((p >= UMBRAL) == (y == 1), pru.sujeto.to_numpy()),
        "rejilla": filas,
    }


def main():
    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    d = d[d.fuente == "uta"]  # con KSS 5: la regresión lo usa
    print("calculando descriptores por ventana...", flush=True)
    v = d.groupby("ventana").apply(descriptores_ventana, include_groups=False)
    v = v.join(d.groupby("ventana")[["particion", "balanceado", "somnolencia"]].first())
    v = v[v.n >= MIN_FOTOGRAMAS]
    v["balanceado"] = v.balanceado.fillna(False).astype(bool)
    partes = {p: v[v.particion == p] for p in ("entrenamiento", "validacion", "prueba")}
    print(", ".join(f"{p} {len(x)} ventanas" for p, x in partes.items()))

    resultado = {}
    for familia in ("regresion", "knn", "svm", "rf"):
        modelo, r = regresion(partes) if familia == "regresion" else clasificador(familia, partes)
        joblib.dump(modelo, DIR_MODELOS / f"clasico_{familia}.joblib")
        resultado[familia] = r

    (config.REPORTES / "clasicos_resultado.json").write_text(json.dumps(resultado, indent=2), encoding="utf8")

    print("\nPrueba (KSS 0 contra KSS 10, mismas ventanas para los 4):")
    print(f"{'modelo':10s} {'exactitud':>9s} {'precision':>9s} {'sensib.':>8s} {'F1':>6s} {'AP':>6s} {'ms/vent':>8s}")
    for familia, r in resultado.items():
        m = r["prueba"]
        ap = r.get("precision_promedio_AP", np.nan)
        print(f"{familia:10s} {m['exactitud']:9.3f} {m['precision']:9.3f} {m['sensibilidad']:8.3f} "
              f"{m['f1']:6.3f} {ap:6.3f} {r['latencia_ms_por_ventana']:8.2f}")
    r = resultado["regresion"]
    print(f"\nRegresión lineal sobre el KSS: MAE {r['mae_kss']:.2f}, R² {r['r2_kss']:.3f}, "
          f"KSS estimado medio por nivel {r['kss_estimado_medio_por_nivel']}")
    print("Guardado:", config.REPORTES / "clasicos_resultado.json")


if __name__ == "__main__":
    main()
