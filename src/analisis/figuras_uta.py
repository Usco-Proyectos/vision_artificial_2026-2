"""Gráficos descriptivos de UTA-RLDD para el modelo matemático (Hito 2).

Calcula los descriptores por ventana que usa M4 (EAR medio, PERCLOS, parpadeos,
MAR máximo) y dibuja cómo cambian según el nivel KSS (0 alerta, 5, 10 somnoliento).

Uso:  python -m analisis.figuras_uta   (desde la carpeta src)
Salida: docs/fuentes/resultados/figuras/ y una tabla con las medianas.
"""
from pathlib import Path

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

import config

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SALIDA = Path(__file__).resolve().parents[2] / "docs" / "fuentes" / "resultados" / "figuras"
UMBRAL_EAR = 0.20          # ojo cerrado si EAR < 0,20 (valor usual en la literatura)
MIN_FOTOGRAMAS = 5         # ventanas con menos fotogramas con rostro no se usan
COLORES = {0: "#86b6ef", 5: "#2a78d6", 10: "#104281"}   # una sola tonalidad, claro -> oscuro
NOMBRES = {0: "KSS 0\n(alerta)", 5: "KSS 5\n(vigilancia baja)", 10: "KSS 10\n(somnoliento)"}
TINTA, TINTA_2, FONDO = "#0b0b0b", "#52514e", "#fcfcfb"


def descriptores_ventana(d):
    d = d.sort_values("fotograma")
    ear = (d["ear_der"] + d["ear_izq"]).to_numpy() / 2
    cerrado = ear < UMBRAL_EAR
    # Un parpadeo = paso de abierto a cerrado entre fotogramas seguidos
    parpadeos = int(np.sum(~cerrado[:-1] & cerrado[1:])) + int(cerrado[0])
    return pd.Series({
        "kss": d["kss"].iloc[0],
        "sujeto": d["sujeto"].iloc[0],
        "n": len(d),
        "ear_medio": ear.mean(),
        "perclos": cerrado.mean(),
        "parpadeos": parpadeos,
        "mar_max": d["mar"].max(),
    })


def caja(ax, datos, titulo, eje_y):
    niveles = [0, 5, 10]
    partes = ax.boxplot([datos[datos.kss == k]["v"] for k in niveles], widths=0.5,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": FONDO, "linewidth": 2},
                        whiskerprops={"color": TINTA_2, "linewidth": 1},
                        capprops={"color": TINTA_2, "linewidth": 1})
    for caja_k, k in zip(partes["boxes"], niveles):
        caja_k.set(facecolor=COLORES[k], edgecolor=FONDO, linewidth=2)
    ax.set_xticks([1, 2, 3], [NOMBRES[k] for k in niveles], fontsize=8, color=TINTA)
    ax.set_title(titulo, fontsize=10, color=TINTA, loc="left")
    ax.set_ylabel(eje_y, fontsize=9, color=TINTA_2)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c2")
    ax.tick_params(colors=TINTA_2, labelsize=8)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda y, _: f"{y:g}".replace(".", ",")))


def main():
    SALIDA.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(config.PROCESADO / "uta.csv")
    v = d.groupby("ventana")[d.columns].apply(descriptores_ventana).reset_index()
    v = v[v.n >= MIN_FOTOGRAMAS]
    v["kss"] = v["kss"].astype(int)

    paneles = [("ear_medio", "a) EAR medio por ventana", "EAR"),
               ("perclos", f"b) PERCLOS por ventana (EAR < {UMBRAL_EAR:.2f})".replace(".", ","), "fracción de fotogramas"),
               ("mar_max", "c) MAR máximo por ventana", "MAR")]
    fig, ejes = plt.subplots(1, 3, figsize=(10, 3.6), facecolor=FONDO)
    for ax, (col, titulo, eje_y) in zip(ejes, paneles):
        ax.set_facecolor(FONDO)
        caja(ax, v.rename(columns={col: "v"}), titulo, eje_y)
    fig.text(0.01, 0.01, f"UTA-RLDD, {v.sujeto.nunique()} sujetos, {len(v)} ventanas de 10 fotogramas a 3 FPS. "
             "Cajas: cuartiles; línea: mediana; bigotes: 1,5 RIC.", fontsize=7, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SALIDA / "fig_descriptores_kss.png", dpi=300)

    resumen = v.groupby("kss")[["ear_medio", "perclos", "parpadeos", "mar_max"]].median().round(3)
    resumen["ventanas"] = v.groupby("kss").size()
    resumen.to_csv(SALIDA / "descriptores_kss_medianas.csv")
    print(resumen.to_string())


if __name__ == "__main__":
    main()
