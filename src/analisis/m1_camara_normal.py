"""¿M1 sirve con cámara normal? Prueba de M1 sobre los recortes de ojo de UTA y YawDD.

M1 se entrenó con MRL, que son imágenes infrarrojas. En la app va a ver recortes de una
cámara normal pasados a gris, como los de UTA y YawDD. Esos recortes no tienen etiqueta de
ojo cerrado, así que se comparan con el EAR, que es una medida geométrica independiente:

- "cerrado seguro": EAR medio < 0,15;   "abierto seguro": EAR medio > 0,25.
  (Los valores intermedios se dejan por fuera porque el EAR ahí es ambiguo.)
- Se mide qué tanto coincide M1 con esa etiqueta débil, la correlación entre la
  probabilidad de M1 y el EAR (debería ser negativa) y, en UTA, qué fracción de fotogramas
  marca M1 como ojo cerrado en cada nivel KSS (debería subir con la somnolencia).

Uso:  python -m analisis.m1_camara_normal   (desde la carpeta src)
Salida (en C:\\datasets\\procesado\\reportes y docs/fuentes/resultados/figuras):
  m1_camara_normal.csv, m1_camara_normal.json, fig_m1_camara_normal.png, fig_m1_ejemplos.png
"""
import json

import cv2
import matplotlib
import matplotlib.ticker
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from analisis.figuras_uta import FONDO, SALIDA, TINTA, TINTA_2
from modelos.entrenar_m1 import DESV, MEDIA, UMBRAL, a_tres_canales, crear_red, metricas

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

EAR_CERRADO, EAR_ABIERTO = 0.15, 0.25
AZUL, NARANJA = "#2a78d6", "#eb6834"
COLORES_KSS = {0: "#86b6ef", 5: "#2a78d6", 10: "#104281"}


class Recortes(Dataset):
    def __init__(self, rutas):
        self.rutas = [str(config.PROCESADO / r) for r in rutas]
        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Lambda(a_tres_canales),
                                      transforms.Normalize(MEDIA, DESV)])

    def __len__(self):
        return len(self.rutas)

    def __getitem__(self, i):
        img = cv2.imread(self.rutas[i], cv2.IMREAD_GRAYSCALE)
        if img.shape != (config.TAM_OJO, config.TAM_OJO):
            img = cv2.resize(img, (config.TAM_OJO, config.TAM_OJO), interpolation=cv2.INTER_AREA)
        return self.tf(img)


@torch.no_grad()
def predecir(red, rutas, disp):
    cargador = DataLoader(Recortes(rutas), batch_size=256, num_workers=2, pin_memory=True)
    salida = []
    for i, x in enumerate(cargador):
        with torch.autocast(disp.type, enabled=disp.type == "cuda"):
            salida.append(torch.sigmoid(red(x.to(disp)).float().squeeze(1)).cpu().numpy())
        if i % 50 == 0:
            print(f"  {i * 256}/{len(rutas)}", flush=True)
    return np.concatenate(salida)


def ejes_limpios(ax):
    ax.set_facecolor(FONDO)
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("#c9c8c2")
    ax.tick_params(colors=TINTA_2, labelsize=8)


def main():
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    red = crear_red().to(disp)
    red.load_state_dict(torch.load(config.RAIZ_DATOS / "modelos" / "m1_mejor.pt",
                                   map_location=disp, weights_only=True))
    red.eval()

    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    d = d[d.fuente.isin(["uta", "yawdd"])].reset_index(drop=True)
    print(f"Recortes: {len(d)} fotogramas x 2 ojos (UTA y YawDD), en {disp}")
    print("ojo derecho:"); d["p_der"] = predecir(red, d["ruta_ojo_der"], disp)
    print("ojo izquierdo:"); d["p_izq"] = predecir(red, d["ruta_ojo_izq"], disp)
    d["p"] = (d.p_der + d.p_izq) / 2          # promedio de los dos ojos, como en el modelo
    d["ear"] = (d.ear_der + d.ear_izq) / 2
    cols = ["id", "fuente", "sujeto", "kss", "bostezo", "ear_der", "ear_izq", "ear",
            "p_der", "p_izq", "p", "ruta_ojo_der"]
    d[cols].to_csv(config.REPORTES / "m1_camara_normal.csv", index=False)

    # Etiqueta débil a partir del EAR (solo los casos claros)
    seguro = d[(d.ear < EAR_CERRADO) | (d.ear > EAR_ABIERTO)].copy()
    seguro["y"] = (seguro.ear < EAR_CERRADO).astype(int)
    resumen = {"umbrales_ear": {"cerrado_si_menor": EAR_CERRADO, "abierto_si_mayor": EAR_ABIERTO}}
    for fuente, g in [("todas", seguro), *seguro.groupby("fuente")]:
        resumen[f"acuerdo_con_ear_{fuente}"] = metricas(g.y.to_numpy(), g.p.to_numpy())
    resumen["correlacion_spearman_p_vs_ear"] = {
        f: round(float(g[["p", "ear"]].corr(method="spearman").iloc[0, 1]), 4)
        for f, g in [("todas", d), *d.groupby("fuente")]}
    resumen["acuerdo_entre_ojos"] = round(float(((d.p_der >= UMBRAL) == (d.p_izq >= UMBRAL)).mean()), 4)
    uta = d[d.fuente == "uta"]
    resumen["uta_fraccion_cerrado_por_kss"] = {
        str(int(k)): round(float((g.p >= UMBRAL).mean()), 4) for k, g in uta.groupby("kss")}
    resumen["fraccion_cerrado_total"] = {f: round(float((g.p >= UMBRAL).mean()), 4)
                                         for f, g in d.groupby("fuente")}
    (config.REPORTES / "m1_camara_normal.json").write_text(json.dumps(resumen, indent=2), encoding="utf8")
    print(json.dumps(resumen, indent=2))

    # Figura: a) probabilidad de M1 según el EAR; b) fracción "cerrado" por KSS en UTA
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6), facecolor=FONDO, gridspec_kw={"width_ratios": [1.6, 1]})
    bins = np.linspace(0, 1, 26)
    for y, nombre, color in [(0, f"abierto según EAR (> {EAR_ABIERTO:.2f})", AZUL),
                             (1, f"cerrado según EAR (< {EAR_CERRADO:.2f})", NARANJA)]:
        v = seguro[seguro.y == y].p
        a.hist(v, bins=bins, weights=np.ones(len(v)) / max(len(v), 1), color=color, alpha=0.75,
               edgecolor=FONDO, linewidth=1, label=nombre.replace(".", ","))
    a.axvline(UMBRAL, color=TINTA_2, linewidth=1, linestyle="--")
    a.set_title("a) Probabilidad de ojo cerrado que da M1", fontsize=10, color=TINTA, loc="left")
    a.set_xlabel("probabilidad de M1 (promedio de los dos ojos)", fontsize=9, color=TINTA_2)
    a.set_ylabel("fracción de fotogramas", fontsize=9, color=TINTA_2)
    a.legend(fontsize=7.5, frameon=False, labelcolor=TINTA)
    ejes_limpios(a)
    niveles = [k for k in (0, 5, 10) if str(k) in resumen["uta_fraccion_cerrado_por_kss"]]
    valores = [resumen["uta_fraccion_cerrado_por_kss"][str(k)] for k in niveles]
    b.bar([f"KSS {k}" for k in niveles], valores, color=[COLORES_KSS[k] for k in niveles], width=0.55)
    for i, v in enumerate(valores):
        b.text(i, v, f"{100 * v:.1f} %".replace(".", ","), ha="center", va="bottom", fontsize=8, color=TINTA)
    b.set_title("b) UTA: fotogramas con ojo cerrado según M1", fontsize=10, color=TINTA, loc="left")
    b.set_ylabel("fracción de fotogramas", fontsize=9, color=TINTA_2)
    ejes_limpios(b)
    for ax in (a, b):
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",")))
    fig.text(0.01, 0.01, f"M1 entrenado con MRL (infrarrojo) y aplicado a recortes de cámara normal de UTA y YawDD "
             f"({len(d)} fotogramas). Etiqueta de referencia: EAR, solo casos claros ({len(seguro)}).",
             fontsize=7, color=TINTA_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(SALIDA / "fig_m1_camara_normal.png", dpi=300)

    # Ejemplos para revisar a ojo: lo más "cerrado", lo más "abierto" y los desacuerdos con el EAR
    rng = np.random.default_rng(config.SEMILLA)
    grupos = [("M1: cerrado (p más alta)", d.nlargest(8, "p")),
              ("M1: abierto (p más baja)", d.nsmallest(8, "p")),
              ("M1 dice cerrado, EAR dice abierto", seguro[(seguro.y == 0) & (seguro.p >= UMBRAL)]),
              ("M1 dice abierto, EAR dice cerrado", seguro[(seguro.y == 1) & (seguro.p < UMBRAL)])]
    fig, ejes = plt.subplots(4, 8, figsize=(10, 5.6), facecolor=FONDO)
    for fila, (titulo, g) in enumerate(grupos):
        if len(g) > 8:
            g = g.iloc[rng.choice(len(g), 8, replace=False)]
        ejes[fila, 0].set_title(titulo, fontsize=8, color=TINTA, loc="left", x=0)
        for j in range(8):
            ax = ejes[fila, j]
            ax.axis("off")
            if j < len(g):
                r = g.iloc[j]
                ax.imshow(cv2.imread(str(config.PROCESADO / r.ruta_ojo_der), cv2.IMREAD_GRAYSCALE), cmap="gray")
                ax.text(0.5, -0.12, f"p={r.p:.2f} EAR={r.ear:.2f}".replace(".", ","), transform=ax.transAxes,
                        ha="center", fontsize=6, color=TINTA_2)
    fig.tight_layout()
    fig.savefig(SALIDA / "fig_m1_ejemplos.png", dpi=200)
    print("Figuras en", SALIDA)


if __name__ == "__main__":
    main()
