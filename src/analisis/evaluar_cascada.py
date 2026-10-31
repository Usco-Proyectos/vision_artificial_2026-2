"""Cascada C1 -> C2 -> C3 sobre las ventanas de UTA-RLDD: ¿qué tan bien decide el sistema completo?

Repite la lógica de back/app/orchestrator.py sin correr ninguna red. Usa lo que ya quedó guardado
al entrenar M3 con --entrada senales:
  - reportes/uta_senales.csv: P(cerrado) de M1 en cada ojo y P(bostezo) de M2, por fotograma.
  - reportes/m3_senales_predicciones.csv: z3 de M3 por ventana (particiones combinador y prueba).

Una ventana (10 fotogramas a 3 FPS, ~3,3 s) queda como "alerta" si en ella se cumple alguna regla:
  C1  ojo cerrado (el más cerrado de los dos >= 0,5) en N_o fotogramas seguidos (N_o = 3 en el prototipo)
  C2  z3 de M3 >= 0,5
  C3  algún fotograma con P(bostezo) >= 0,5
La regla que se reporta es la de mayor prioridad que se cumpla. Se compara con la etiqueta de la
ventana: KSS 10 = somnoliento (1), KSS 0 = alerta (0). KSS 5 no entra, igual que en M3.

Además: cada regla sola y en parejas (para ver cuánto aporta cada una) y N_o de 1 a 5.

Con --con-c4 se agrega la regla C4 (Random Forest de M4 >= 0,5, de reportes/m4_predicciones.csv),
en prioridad C1 -> C2 -> C4 -> C3, para ver si vale la pena ponerla en el prototipo.

Uso:  python -m analisis.evaluar_cascada [--con-c4]   (desde la carpeta src)
Salida: reportes/cascada_resultado.json (o cascada_c4_resultado.json) y un resumen en pantalla.
"""
import argparse
import itertools
import json

import numpy as np
import pandas as pd

import config
from modelos.entrenar_m1 import metricas

UMBRAL = 0.5
N_O = 3
REGLAS = ("C1", "C2", "C3")
REGLAS_C4 = ("C1", "C2", "C4", "C3")  # en orden de prioridad


def racha_max(fotogramas, cerrado):
    """Mayor número de fotogramas seguidos con ojo cerrado. Si falta un fotograma (sin rostro),
    la racha se corta, igual que en el prototipo (sin rostro, o_t = 0)."""
    paso = np.median(np.diff(fotogramas)) if len(fotogramas) > 1 else 1
    k = mejor = 0
    for i, c in enumerate(cerrado):
        seguido = i > 0 and fotogramas[i] - fotogramas[i - 1] <= 1.5 * paso
        k = (k + 1 if seguido else 1) if c else 0
        mejor = max(mejor, k)
    return mejor


def ventanas(particiones):
    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False,
                    usecols=["id", "fuente", "sujeto", "particion", "ventana", "fotograma", "somnolencia"])
    d = d[(d.fuente == "uta") & d.somnolencia.notna() & d.particion.isin(particiones)]
    s = pd.read_csv(config.REPORTES / "uta_senales.csv")
    z3 = pd.read_csv(config.REPORTES / "m3_senales_predicciones.csv")
    d = d.merge(s, on="id")
    if d.empty:
        raise SystemExit("uta_senales.csv no coincide con dataset.csv (ids distintos). ¿Son de la misma consolidación?")

    filas = []
    for w, v in d.sort_values("fotograma").groupby("ventana"):
        cerrado = v[["p_ojo_der", "p_ojo_izq"]].max(axis=1).to_numpy() >= UMBRAL
        filas.append({"ventana": w, "sujeto": v.sujeto.iat[0], "particion": v.particion.iat[0],
                      "y": v.somnolencia.iat[0], "n_w": len(v),
                      "racha": racha_max(v.fotograma.to_numpy(), cerrado),
                      "p_boca_max": v.p_boca.max()})
    w = pd.DataFrame(filas).merge(z3[["ventana", "y", "z3"]], on="ventana", suffixes=("", "_m3"))
    assert (w.y == w.y_m3).all(), "la etiqueta de la ventana no coincide con la de las predicciones de M3"
    ruta_z4 = config.REPORTES / "m4_predicciones.csv"
    if ruta_z4.exists():
        w = w.merge(pd.read_csv(ruta_z4)[["ventana", "z4"]], on="ventana", how="left")
    return w


def disparos(w, n_o):
    r = {"C1": w.racha >= n_o, "C2": w.z3 >= UMBRAL, "C3": w.p_boca_max >= UMBRAL}
    if "z4" in w:
        r["C4"] = w.z4 >= UMBRAL
    return r


def evaluar(w, reglas, n_o=N_O):
    r = disparos(w, n_o)
    alerta = np.zeros(len(w), bool)
    for c in reglas:
        alerta |= r[c].to_numpy()
    return metricas(w.y.to_numpy(), alerta.astype(float))


def regla_reportada(w, reglas, n_o=N_O):
    r = disparos(w, n_o)
    return np.select([r[c] for c in reglas], list(reglas), "BASE")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--con-c4", action="store_true", help="agrega la regla C4 con el Random Forest de M4")
    con_c4 = ap.parse_args().con_c4
    REGLAS_USO = REGLAS_C4 if con_c4 else REGLAS
    w_todas = ventanas(["prueba", "combinador"])
    if con_c4 and ("z4" not in w_todas or w_todas.z4.isna().any()):
        raise SystemExit("Falta reportes/m4_predicciones.csv o no tiene todas las ventanas.")
    resultado = {"umbral": UMBRAL, "n_o": N_O, "reglas": list(REGLAS_USO), "particiones": {}}

    for nombre, parts in [("prueba", ["prueba"]), ("reservada", ["combinador"]),
                          ("prueba+reservada", ["prueba", "combinador"])]:
        w = w_todas[w_todas.particion.isin(parts)]
        reglas = pd.crosstab(regla_reportada(w, REGLAS_USO), w.y.map({0.0: "alerta (KSS 0)", 1.0: "somnoliento (KSS 10)"}))
        acierto = (regla_reportada(w, REGLAS_USO) != "BASE") == (w.y == 1)
        combos = {"+".join(c): evaluar(w, c) for k in range(1, len(REGLAS_USO) + 1)
                  for c in itertools.combinations(REGLAS_USO, k)}
        resultado["particiones"][nombre] = {
            "ventanas": len(w), "sujetos": int(w.sujeto.nunique()),
            "somnolientas": int(w.y.sum()),
            "cascada": evaluar(w, REGLAS_USO),
            "regla_reportada": reglas.to_dict(),
            "exactitud_por_sujeto": acierto.groupby(w.sujeto.to_numpy()).mean().round(4).to_dict(),
            "por_combinacion_de_reglas": combos,
            "cascada_segun_n_o": {n: evaluar(w, REGLAS_USO, n) for n in range(1, 6)},
        }

        m = resultado["particiones"][nombre]["cascada"]
        print(f"\n=== {nombre}: {len(w)} ventanas, {w.sujeto.nunique()} sujetos, {int(w.y.sum())} somnolientas ===")
        print(f"cascada completa: exactitud {m['exactitud']:.3f}  precisión {m['precision']:.3f}  "
              f"sensibilidad {m['sensibilidad']:.3f}  F1 {m['f1']:.3f}  (VP {m['VP']} VN {m['VN']} FP {m['FP']} FN {m['FN']})")
        print("regla reportada por clase:\n" + reglas.to_string())
        print("reglas por separado y combinadas:")
        for c, x in combos.items():
            print(f"  {c:12s} exactitud {x['exactitud']:.3f}  sensibilidad {x['sensibilidad']:.3f}  "
                  f"falsas alarmas {x['FP']}/{x['FP'] + x['VN']}  F1 {x['f1']:.3f}")
        print("cascada según N_o (fotogramas seguidos para C1):")
        for n, x in resultado["particiones"][nombre]["cascada_segun_n_o"].items():
            print(f"  N_o={n}  exactitud {x['exactitud']:.3f}  sensibilidad {x['sensibilidad']:.3f}  F1 {x['f1']:.3f}")

    salida = config.REPORTES / ("cascada_c4_resultado.json" if con_c4 else "cascada_resultado.json")
    salida.write_text(json.dumps(resultado, indent=2, ensure_ascii=False), encoding="utf8")
    print("\nGuardado:", salida)


# Chequeo rápido de la racha: hueco (sin rostro) corta, ojo abierto corta
assert racha_max(np.array([0, 10, 20, 30]), np.array([1, 1, 1, 0], bool)) == 3
assert racha_max(np.array([0, 10, 30, 40]), np.array([1, 1, 1, 1], bool)) == 2
assert racha_max(np.array([0, 10, 20]), np.array([1, 0, 1], bool)) == 1

if __name__ == "__main__":
    main()
