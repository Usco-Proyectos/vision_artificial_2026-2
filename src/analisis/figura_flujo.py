"""Figura 2 del artículo: flujo del sistema, del fotograma a la alerta (C1 y votación de M1-M4).

Uso:  python -m analisis.figura_flujo   (desde la carpeta src)
"""
import matplotlib

from analisis.figuras_uta import FONDO, SALIDA, TINTA, TINTA_2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

AZUL, NARANJA, GRIS = "#2a78d6", "#eb6834", "#c9c8c2"
CLARO = {"azul": "#e8f1fc", "naranja": "#fdeee7", "gris": "#f3f3f1"}


def caja(ax, x, y, w, h, texto, borde, relleno, punteada=False, negrita=False):
    """Caja centrada en (x, y); devuelve sus bordes izquierdo y derecho para las flechas."""
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.008,rounding_size=0.015",
                                facecolor=relleno, edgecolor=borde, linewidth=1.4,
                                linestyle="--" if punteada else "-"))
    ax.text(x, y, texto, ha="center", va="center", fontsize=8, color=TINTA,
            fontweight="bold" if negrita else "normal", linespacing=1.35)
    return x - w / 2, x + w / 2


def flecha(ax, x0, y0, x1, y1, punteada=False):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops={"arrowstyle": "-|>", "color": TINTA_2, "linewidth": 1.1,
                            "linestyle": "--" if punteada else "-", "shrinkA": 0, "shrinkB": 0})


def main():
    fig, ax = plt.subplots(figsize=(11, 4.2), facecolor=FONDO)
    ax.set_xlim(-0.01, 1.01); ax.set_ylim(0, 1); ax.axis("off")

    w, h = 0.115, 0.2
    _, d1 = caja(ax, 0.065, 0.5, w, h, "Fotograma\n(cámara del\ncelular)", GRIS, CLARO["gris"])
    i2, d2 = caja(ax, 0.215, 0.5, w, h, "MediaPipe\n468 puntos\ndel rostro", GRIS, CLARO["gris"])
    i3, d3 = caja(ax, 0.365, 0.5, w, h + 0.06, "Recortes de\nojos y boca\n96 × 96, gris\n+ EAR y MAR", GRIS, CLARO["gris"])
    flecha(ax, d1, 0.5, i2, 0.5)
    flecha(ax, d2, 0.5, i3, 0.5)

    # Modelos: M1 y M2 en cada fotograma; M3 y M4 por ventana
    wm, hm = 0.16, 0.15
    modelos = [
        (0.86, "M1 · MobileNetV3-Small\nojo cerrado (cada ojo)", False),
        (0.63, "M2 · MobileNetV3-Small\nbostezo", False),
        (0.38, "M3 · LSTM sobre señales\nsomnolencia por ventana", False),
        (0.13, "M4 · Random Forest\nsomnolencia por ventana", False),
    ]
    bordes_m = {}
    for y, texto, punteada in modelos:
        im, dm = caja(ax, 0.555, y, wm, hm, texto, GRIS if punteada else AZUL,
                      CLARO["gris"] if punteada else CLARO["azul"], punteada=punteada)
        bordes_m[y] = dm
        flecha(ax, d3, 0.5, im, y, punteada=punteada)
    ax.text(0.555, 0.995, "cada fotograma", ha="center", va="top", fontsize=7.5, color=TINTA_2, style="italic")
    ax.text(0.555, 0.505, "por ventana", ha="center", va="center", fontsize=7.5, color=TINTA_2, style="italic")

    # C1 va primero y aparte; si no se cumple, alerta con al menos 2 votos de los 4 modelos
    ic, dc = caja(ax, 0.79, 0.6, 0.165, 0.62,
                  "Regla fija y votación\n\n"
                  "C1  ojo cerrado ≥ 3\n       fotogramas seguidos\n"
                  "C2  al menos 2 de 4\n       votos (M1, M2,\n       M3 y M4)\n"
                  "BASE  sin alerta",
                  NARANJA, CLARO["naranja"])
    for y in (0.86, 0.63, 0.38, 0.13):
        flecha(ax, bordes_m[y], y, ic, min(max(y, 0.33), 0.87))

    ia, _ = caja(ax, 0.94, 0.6, 0.1, 0.2, "Alerta\nen pantalla\n+ vibración", NARANJA, CLARO["naranja"], negrita=True)
    flecha(ax, dc, 0.6, ia, 0.6)

    # Dónde corre cada parte
    ax.text(0.065, 0.3, "celular", ha="center", va="center", fontsize=7.5, color=TINTA_2, style="italic")
    ax.text(0.94, 0.42, "celular", ha="center", va="center", fontsize=7.5, color=TINTA_2, style="italic")
    ax.text(0.5, 0.005, "servidor (portátil con GPU)", ha="center", va="bottom", fontsize=7.5, color=TINTA_2, style="italic")

    fig.tight_layout()
    SALIDA.mkdir(parents=True, exist_ok=True)
    fig.savefig(SALIDA / "fig_flujo_cascada.png", dpi=300)
    print("Guardada:", SALIDA / "fig_flujo_cascada.png")


if __name__ == "__main__":
    main()
