"""Gráfico y tabla del dataset consolidado (análisis exploratorio, Hito 2).

Muestra, para cada fuente, cuántos registros quedaron en cada partición y cómo se
reparten las clases. La clase "señal" (ojo cerrado, bostezo, somnoliento) va en naranja.

Uso:  python -m analisis.figuras_dataset   (desde la carpeta src)
Salida: docs/fuentes/resultados/figuras/fig_dataset_particiones.png y dataset_resumen.csv
"""
import matplotlib
import pandas as pd

import config
from analisis.figuras_uta import FONDO, SALIDA, TINTA, TINTA_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#c9c8c2"
# (fuente, título, columna, [(valor, nombre, color)])
FUENTES = [
    ("mrl", "a) MRL Eye: estado del ojo (M1)", "ojo",
     [(1, "abierto", AZUL), (0, "cerrado", NARANJA)]),
    ("yawdd", "b) YawDD: bostezo (M2)", "bostezo",
     [(0, "no bostezo", AZUL), (1, "bostezo", NARANJA)]),
    ("uta", "c) UTA-RLDD: somnolencia (M3, M4, M5)", "kss",
     [(0, "alerta (KSS 0)", AZUL), (10, "somnoliento (KSS 10)", NARANJA),
      (5, "KSS 5 (no entra al binario)", GRIS)]),
]
ORDEN = ["entrenamiento", "validacion", "combinador", "prueba"]
NOMBRE_PART = {"entrenamiento": "Entrenamiento", "validacion": "Validación",
               "combinador": "Combinador (M5)", "prueba": "Prueba"}


def miles(x):
    return f"{x:,.0f}".replace(",", " ")


def main():
    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    fig, ejes = plt.subplots(3, 1, figsize=(8, 7.2), facecolor=FONDO)
    filas_tabla = []
    for ax, (fuente, titulo, col, clases) in zip(ejes, FUENTES):
        x = d[d.fuente == fuente]
        partes = [p for p in ORDEN if p in set(x.particion)][::-1]
        izquierda = pd.Series(0.0, index=partes)
        for valor, nombre, color in clases:
            n = x[x[col] == valor].groupby("particion").size().reindex(partes, fill_value=0)
            ax.barh([NOMBRE_PART[p] for p in partes], n, left=izquierda, color=color,
                    edgecolor=FONDO, linewidth=2, height=0.6, label=nombre)
            izquierda += n
        for i, p in enumerate(partes):
            xp = x[x.particion == p]
            ax.text(izquierda[p] + x.shape[0] * 0.01, i,
                    f"{miles(len(xp))}  ({xp.sujeto.nunique()} sujetos)",
                    va="center", fontsize=7.5, color=TINTA_2)
            filas_tabla.append({"fuente": fuente, "particion": p, "registros": len(xp),
                                "sujetos": xp.sujeto.nunique(),
                                **{nombre: int((xp[col] == valor).sum()) for valor, nombre, _ in clases}})
        ax.set_title(f"{titulo}: {miles(len(x))} registros, {x.sujeto.nunique()} sujetos",
                     fontsize=9.5, color=TINTA, loc="left")
        ax.set_facecolor(FONDO)
        ax.set_xlim(0, izquierda.max() * 1.32)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: miles(v)))
        ax.grid(axis="x", color="#e6e5e0", linewidth=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color("#c9c8c2")
        ax.tick_params(colors=TINTA_2, labelsize=8)
        ax.legend(loc="lower right", fontsize=7.5, frameon=False, labelcolor=TINTA)
    fig.text(0.01, 0.005, f"Dataset consolidado: {miles(len(d))} registros, {d.sujeto.nunique()} sujetos. "
             "Las particiones son por sujeto: ninguna persona aparece en dos.", fontsize=7, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    fig.savefig(SALIDA / "fig_dataset_particiones.png", dpi=300)
    tabla = pd.DataFrame(filas_tabla).fillna(0)
    tabla.to_csv(SALIDA / "dataset_resumen.csv", index=False)
    print(tabla.to_string(index=False))


if __name__ == "__main__":
    main()
