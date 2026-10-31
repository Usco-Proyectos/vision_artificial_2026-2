"""Ensemble de M1, M2, M3 y M4 por ventana de UTA-RLDD, comparado con la cascada actual.

Usa las mismas ventanas y salidas que evaluar_cascada (no corre ninguna red). Por ventana, cada
modelo da un voto:
  M1  ojo cerrado en N_o = 3 fotogramas seguidos      M2  algún fotograma con P(bostezo) >= 0,5
  M3  z3 >= 0,5                                       M4  z4 >= 0,5
Se prueban:
  - votación: alerta si hay al menos k votos (k = 1 es la cascada con C4)
  - stacking: regresión logística sobre [racha de M1, P(bostezo) máx. de M2, z3, z4]
  - stacking + C1: el ojo cerrado sigue siendo regla fija y el stacking decide lo demás
  - C1 + votos >= 2: lo que quedó en el prototipo (back/app/orchestrator.py)
k (por exactitud) y el stacking se ajustan en la partición combinador (reservada); prueba solo se usa para reportar.

Uso:  python -m analisis.evaluar_ensemble   (desde la carpeta src)
Salida: reportes/ensemble_resultado.json y un resumen en pantalla.
"""
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from analisis.evaluar_cascada import N_O, UMBRAL, disparos, ventanas
from modelos.entrenar_m1 import metricas

ENTRADAS = ["racha", "p_boca_max", "z3", "z4"]


def votos(w):
    r = disparos(w, N_O)
    return (r["C1"].astype(int) + r["C2"] + r["C3"] + r["C4"]).to_numpy()


def evaluar(y, alerta):
    m = metricas(y, alerta.astype(float))
    m["falsas_alarmas"] = m["FP"] / max(m["FP"] + m["VN"], 1)  # ventanas KSS 0 con alerta
    return m


def reglas(w, stacking):
    c1 = (w.racha >= N_O).to_numpy()
    p = stacking.predict_proba(w[ENTRADAS])[:, 1] >= UMBRAL
    n = votos(w)
    assert n.min() >= 0 and n.max() <= 4
    r = {"cascada actual (C1|C2|C3)": c1 | (w.z3 >= UMBRAL).to_numpy() | (w.p_boca_max >= UMBRAL).to_numpy()}
    r |= {f"votos >= {k}": n >= k for k in range(1, 5)}
    r["C1 + votos >= 2 (prototipo)"] = c1 | (n >= 2)
    r["stacking"] = p
    r["stacking + C1"] = p | c1
    return r


def main():
    w = ventanas(["prueba", "combinador"])
    if "z4" not in w or w.z4.isna().any():
        raise SystemExit("Falta reportes/m4_predicciones.csv o no tiene todas las ventanas.")
    comb, pru = w[w.particion == "combinador"], w[w.particion == "prueba"]
    stacking = make_pipeline(StandardScaler(), LogisticRegression()).fit(comb[ENTRADAS], comb.y)
    pesos = dict(zip(ENTRADAS, stacking[-1].coef_[0].round(3).tolist()))

    resultado = {"pesos_stacking_estandarizados": pesos}
    for nombre, parte in (("reservada (ajuste)", comb), ("prueba", pru)):
        r = {k: evaluar(parte.y.to_numpy(), a) for k, a in reglas(parte, stacking).items()}
        resultado[nombre] = r
        print(f"\n=== {nombre}: {len(parte)} ventanas, {parte.sujeto.nunique()} sujetos ===")
        print(f"{'regla':26s} {'exactitud':>9s} {'precision':>9s} {'sensib.':>8s} {'F1':>6s} {'falsas alarmas':>15s}")
        for k, m in r.items():
            print(f"{k:26s} {m['exactitud']:9.3f} {m['precision']:9.3f} {m['sensibilidad']:8.3f} "
                  f"{m['f1']:6.3f} {m['falsas_alarmas']:15.1%}")

    res = resultado["reservada (ajuste)"]
    # exactitud y no F1: con F1 gana k = 1, que da alerta en más de la mitad de las ventanas KSS 0
    k = max((x for x in res if x.startswith("votos")), key=lambda x: res[x]["exactitud"])
    resultado["votacion_elegida_en_reservada"] = k
    print(f"\nVotación elegida en reservada (mejor exactitud): {k}")
    print("Pesos del stacking (entradas estandarizadas):", pesos)
    print("Ojo: el stacking en 'reservada' está medido sobre los mismos datos con que se ajustó.")
    salida = config.REPORTES / "ensemble_resultado.json"
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf8")
    print("Guardado:", salida)


if __name__ == "__main__":
    main()
