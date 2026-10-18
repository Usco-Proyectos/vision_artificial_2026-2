"""Gráficos de resultados de M1 (Hito 2, simulación de resultados).

1. fig_m1_entrenamiento.png: a) pérdida de entrenamiento y validación por época,
   b) F1 de validación por época (con la mejor época marcada), c) matriz de confusión en prueba.
2. fig_m1_fuga.png + m1_fuga.csv: partición por sujeto vs. partición aleatoria
   (solo si ya se corrió `python -m modelos.entrenar_m1 --particion aleatoria`).

Uso:  python -m analisis.figuras_m1   (desde la carpeta src)
"""
import json

import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd

import config
from analisis.figuras_uta import FONDO, SALIDA, TINTA, TINTA_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AZUL, NARANJA = "#2a78d6", "#eb6834"
RAMPA = matplotlib.colors.LinearSegmentedColormap.from_list("azul", ["#f5f9fe", "#86b6ef", "#2a78d6", "#104281"])
coma = matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def ejes_limpios(ax, grilla="y"):
    ax.set_facecolor(FONDO)
    if grilla:
        ax.grid(axis=grilla, color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c2")
    ax.tick_params(colors=TINTA_2, labelsize=8)


def leer(nombre):
    h = config.REPORTES / f"{nombre}_historial.csv"
    r = config.REPORTES / f"{nombre}_resultado.json"
    if not (h.exists() and r.exists()):
        return None, None
    return pd.read_csv(h), json.loads(r.read_text(encoding="utf8"))


def figura_entrenamiento(hist, res):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(11, 3.5), facecolor=FONDO,
                                  gridspec_kw={"width_ratios": [1.2, 1.2, 0.9]})
    for col, nombre, color in [("perdida_ent", "entrenamiento", AZUL), ("perdida_val", "validación", NARANJA)]:
        a.plot(hist.epoca, hist[col], color=color, linewidth=2, marker="o", markersize=4, label=nombre)
        a.text(hist.epoca.iloc[-1] + 0.15, hist[col].iloc[-1], nombre, fontsize=7.5, color=TINTA_2, va="center")
    a.set_title("a) Pérdida por época", fontsize=10, color=TINTA, loc="left")
    a.set_xlabel("época", fontsize=9, color=TINTA_2); a.set_ylabel("entropía cruzada binaria", fontsize=9, color=TINTA_2)
    a.legend(fontsize=7.5, frameon=False, labelcolor=TINTA)

    mejor = hist.loc[hist.f1_val.idxmax()]
    b.plot(hist.epoca, hist.f1_val, color=AZUL, linewidth=2, marker="o", markersize=4)
    b.scatter([mejor.epoca], [mejor.f1_val], s=70, color=NARANJA, zorder=3, edgecolor=FONDO, linewidth=2)
    b.annotate(f"mejor: época {int(mejor.epoca)}\nF1 = {mejor.f1_val:.3f}".replace(".", ","),
               (mejor.epoca, mejor.f1_val), xytext=(8, -28), textcoords="offset points", fontsize=7.5, color=TINTA)
    b.set_title("b) F1 en validación (se guarda el mejor)", fontsize=10, color=TINTA, loc="left")
    b.set_xlabel("época", fontsize=9, color=TINTA_2); b.set_ylabel("F1", fontsize=9, color=TINTA_2)
    for ax in (a, b):
        ejes_limpios(ax)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        ax.yaxis.set_major_formatter(coma)

    p = res["prueba"]
    matriz = np.array([[p["VN"], p["FP"]], [p["FN"], p["VP"]]])
    filas = matriz / matriz.sum(axis=1, keepdims=True)
    c.imshow(filas, cmap=RAMPA, vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            c.text(j, i, f"{matriz[i, j]:,}".replace(",", " ") + f"\n{100 * filas[i, j]:.1f} %".replace(".", ","),
                   ha="center", va="center", fontsize=8.5, color="#ffffff" if filas[i, j] > 0.5 else TINTA)
    c.set_xticks([0, 1], ["abierto", "cerrado"]); c.set_yticks([0, 1], ["abierto", "cerrado"])
    c.set_xlabel("predicción de M1", fontsize=9, color=TINTA_2); c.set_ylabel("etiqueta real", fontsize=9, color=TINTA_2)
    c.set_title("c) Matriz de confusión (prueba)", fontsize=10, color=TINTA, loc="left")
    ejes_limpios(c, grilla=None)
    fig.text(0.01, 0.01, f"M1 (MobileNetV3-Small, MRL Eye). Prueba: {p['n']:,} imágenes de sujetos no vistos; ".replace(",", " ")
             + f"exactitud {100 * p['exactitud']:.2f} %, F1 {p['f1']:.3f}.".replace(".", ","), fontsize=7, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SALIDA / "fig_m1_entrenamiento.png", dpi=300)


def figura_fuga(res_s, res_a):
    nombres = ["exactitud", "precision", "sensibilidad", "f1"]
    etiquetas = ["Exactitud", "Precisión", "Sensibilidad", "F1"]
    tabla = pd.DataFrame({"metrica": etiquetas,
                          "por_sujeto": [res_s["prueba"][m] for m in nombres],
                          "aleatoria": [res_a["prueba"][m] for m in nombres]})
    tabla["diferencia_puntos"] = 100 * (tabla.aleatoria - tabla.por_sujeto)
    tabla.round(4).to_csv(SALIDA / "m1_fuga.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor=FONDO)
    x = np.arange(len(nombres))
    for desp, col, nombre, color in [(-0.19, "por_sujeto", "partición por sujeto (real)", AZUL),
                                     (0.19, "aleatoria", "partición aleatoria (con fuga)", NARANJA)]:
        ax.bar(x + desp, 100 * tabla[col], width=0.36, color=color, label=nombre, edgecolor=FONDO, linewidth=2)
        for xi, v in zip(x + desp, tabla[col]):
            ax.text(xi, 100 * v + 0.15, f"{100 * v:.1f}".replace(".", ","), ha="center", va="bottom",
                    fontsize=7.5, color=TINTA)
    minimo = 100 * min(tabla.por_sujeto.min(), tabla.aleatoria.min())
    ax.set_ylim(max(0, minimo - 5), 100.8)
    ax.set_xticks(x, etiquetas)
    ax.set_ylabel("% en prueba (eje recortado)", fontsize=9, color=TINTA_2)
    ax.set_title("M1: efecto de mezclar sujetos entre entrenamiento y prueba", fontsize=10, color=TINTA, loc="left")
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TINTA, loc="lower right")
    ejes_limpios(ax)
    ax.yaxis.set_major_formatter(coma)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_m1_fuga.png", dpi=300)
    print(tabla.round(4).to_string(index=False))


def main():
    hist, res = leer("m1")
    if hist is None:
        raise SystemExit("Falta el entrenamiento de M1 (m1_historial.csv / m1_resultado.json)")
    figura_entrenamiento(hist, res)
    print("Listo: fig_m1_entrenamiento.png")
    _, res_a = leer("m1_aleatoria")
    if res_a is None:
        print("Todavía no está m1_aleatoria: corre `python -m modelos.entrenar_m1 --particion aleatoria`")
    else:
        figura_fuga(res, res_a)
        print("Listo: fig_m1_fuga.png y m1_fuga.csv")


if __name__ == "__main__":
    main()
