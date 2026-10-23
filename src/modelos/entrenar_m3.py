"""M3: somnolencia con MobileNetV3-Small + LSTM sobre la ventana del rostro de UTA-RLDD (ecs. 18 y 19).

- Una ventana tiene hasta T = 10 recortes del rostro (160x160, gris) en orden. Solo se usan
  ventanas con n_w >= 5 fotogramas con rostro, y solo KSS 0 (alerta, y = 0) y KSS 10 (y = 1).
- La MobileNetV3-Small (sin su capa final, d = 1024) saca un vector por fotograma; la LSTM
  recorre solo los n_w fotogramas reales y su último estado oculto da la probabilidad.
- Mismo bucle que M1 (Adam, η = 1e-3, η a la mitad, parada temprana por F1, reanudable),
  pero con lotes de 16 ventanas (160 imágenes) para que quepa en la GPU de 6 GB.
- El aumento de datos se aplica igual a los 10 fotogramas de la ventana, para no meter
  saltos falsos entre un fotograma y el siguiente.
- Al final guarda las predicciones de las ventanas del combinador y de prueba: M5 las necesita.
- --entrada senales: con el rostro la red memorizaba a las personas (exactitud 0,5 en casi
  todos los sujetos de prueba = la misma respuesta para todas sus ventanas). Con esta opción la
  LSTM no ve la cara: por fotograma recibe s_t = [P(cerrado) de M1 en cada ojo, P(bostezo) de
  M2, EAR de cada ojo, MAR]. M1 y M2 se entrenaron con otras personas (MRL y YawDD), así que no
  tienen cómo memorizar a la gente de UTA. Las salidas de M1 y M2 sobre todos los fotogramas de
  UTA se calculan una vez y quedan en reportes/uta_senales.csv (M5 las vuelve a usar en z1 y z2).

Uso (desde la carpeta src):
    python -m modelos.entrenar_m3 --rapido                 # prueba corta (1 época, pocas ventanas)
    python -m modelos.entrenar_m3                          # partición por sujeto (la real)
    python -m modelos.entrenar_m3 --particion aleatoria    # solo para medir la fuga
    python -m modelos.entrenar_m3 --congelar               # CNN fija, solo aprende la LSTM
    python -m modelos.entrenar_m3 --entrada senales        # LSTM sobre las salidas de M1, M2, EAR y MAR
    python -m modelos.entrenar_m3 --lote 8                 # si se queda sin memoria la GPU
"""
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import config
from modelos.entrenar_m1 import (DESV, DIR_MODELOS, EPOCAS, MEDIA, Recortes, crear_red, entrenar, evaluar,
                                 metricas)

T = config.UTA_FOTOGRAMAS_CLIP
MIN_FOTOGRAMAS = 5
OCULTO = 128
SENALES = ["p_ojo_der", "p_ojo_izq", "p_boca", "ear_der", "ear_izq", "mar"]


@torch.no_grad()
def probabilidades(pesos, rutas, disp):
    """Salida de M1 o M2 (según `pesos`) para una lista de recortes."""
    red = crear_red().to(disp)
    red.load_state_dict(torch.load(DIR_MODELOS / pesos, map_location=disp, weights_only=True))
    red.eval()
    c = DataLoader(Recortes(rutas, np.zeros(len(rutas)), False), batch_size=256, num_workers=2)
    return np.concatenate([torch.sigmoid(red(x.to(disp)).squeeze(1).float()).cpu().numpy() for x, _ in c])


def senales_uta(d, disp):
    """P(cerrado) de M1 en cada ojo y P(bostezo) de M2 en cada fotograma de UTA (queda en caché)."""
    ruta = config.REPORTES / "uta_senales.csv"
    if not ruta.exists():
        print("calculando M1 y M2 sobre los fotogramas de UTA (solo la primera vez)...", flush=True)
        pd.DataFrame({"id": d.id.to_numpy(),
                      "p_ojo_der": probabilidades("m1_mejor.pt", d.ruta_ojo_der, disp),
                      "p_ojo_izq": probabilidades("m1_mejor.pt", d.ruta_ojo_izq, disp),
                      "p_boca": probabilidades("m2_mejor.pt", d.ruta_boca, disp)}).to_csv(ruta, index=False)
    return pd.read_csv(ruta)


class Senales(Dataset):
    """Igual que Ventanas, pero cada fotograma es el vector s_t de 6 números: T x 6."""

    def __init__(self, df, aumentar=False):
        g = df.sort_values("fotograma").groupby("ventana", sort=True)
        self.ventanas = list(g.groups)
        self.x = [torch.tensor(v[SENALES].to_numpy(), dtype=torch.float32) for _, v in g]
        self.y = g["somnolencia"].first().to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.x[i]
        return torch.cat([x, torch.full((T - len(x), x.shape[1]), float("nan"))]), self.y[i]


class Ventanas(Dataset):
    """Cada elemento es una ventana: tensor T x 3 x 160 x 160. Los fotogramas que faltan
    (sin rostro) van al final llenos de NaN, así la red sabe cuántos son reales sin tener
    que cambiar el bucle de entrenamiento de M1."""

    def __init__(self, df, aumentar):
        g = df.sort_values("fotograma").groupby("ventana", sort=True)
        self.ventanas = list(g.groups)
        self.rutas = [[str(config.PROCESADO / r) for r in x] for x in g["ruta_rostro"].agg(list)]
        self.y = g["somnolencia"].first().to_numpy(dtype=np.float32)
        self.aumento = transforms.Compose([transforms.RandomHorizontalFlip(),
                                           transforms.RandomRotation(10),
                                           transforms.ColorJitter(brightness=0.3, contrast=0.3)]) if aumentar else None
        self.norm = transforms.Normalize(MEDIA, DESV)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        imgs = []
        for r in self.rutas[i]:
            img = cv2.imread(r, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"No está el recorte {r}. ¿Están los recortes en {config.RECORTES}?")
            imgs.append(torch.from_numpy(img))
        x = torch.stack(imgs).unsqueeze(1).float() / 255  # n x 1 x 160 x 160 en [0, 1]
        if self.aumento:
            x = self.aumento(x)  # un solo sorteo para toda la ventana
        x = self.norm(x.expand(-1, 3, -1, -1))
        relleno = torch.full((T - len(x), *x.shape[1:]), float("nan"))
        return torch.cat([x, relleno]), self.y[i]


class CnnLstm(nn.Module):
    def __init__(self, congelar=False, senales=False):
        super().__init__()
        if senales:  # las CNN (M1 y M2) ya corrieron antes; aquí solo se estandariza s_t
            self.cnn = None
            self.register_buffer("media", torch.zeros(len(SENALES)))  # quedan guardadas dentro del .pt
            self.register_buffer("desv", torch.ones(len(SENALES)))
        else:
            self.cnn = crear_red()
            self.cnn.classifier[3] = nn.Identity()  # sin la capa final: e_t ∈ R^1024 (ec. 18)
            # Congelada, la CNN queda con los pesos de ImageNet y solo aprende la LSTM: menos
            # parámetros para memorizar caras (sin congelar se sobreajustó desde la época 2)
            self.cnn.requires_grad_(not congelar)
        self.congelada = congelar
        self.forma = (len(SENALES),) if senales else (3, config.TAM_ROSTRO, config.TAM_ROSTRO)
        self.lstm = nn.LSTM(len(SENALES) if senales else 1024, OCULTO, batch_first=True)
        self.salida = nn.Linear(OCULTO, 1)  # ec. 19

    def train(self, modo=True):
        super().train(modo)
        if self.congelada:
            self.cnn.eval()  # que la normalización por lotes no cambie sus promedios
        return self

    def forward(self, x):
        b = x.shape[0]
        reales = ~torch.isnan(x.flatten(2)[:, :, 0])  # b x T; los reales van primero
        # solo los fotogramas con rostro pasan por la CNN (o se estandarizan, con señales)
        f = (x[reales] - self.media) / self.desv if self.cnn is None else self.cnn(x[reales])
        e = f.new_zeros(b, T, f.shape[1])
        e[reales] = f
        sec = pack_padded_sequence(e, reales.sum(1).cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(sec)
        return self.salida(h[-1])                # b x 1 (logit)


@torch.no_grad()
def medir_latencia(red, disp, repeticiones=100):
    """Con --entrada senales no incluye el tiempo de M1 y M2 (ese se mide con ellos)."""
    red.eval()
    x = torch.randn(1, T, *red.forma, device=disp)
    for _ in range(10):  # calentamiento, no se cuenta
        red(x)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        red(x)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    return {"latencia_ms_por_ventana": (time.perf_counter() - t0) / repeticiones * 1000}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", choices=["sujeto", "aleatoria"], default="sujeto")
    ap.add_argument("--lote", type=int, default=16, help="ventanas por lote")
    ap.add_argument("--hilos", type=int, default=2, help="procesos que leen imágenes")
    ap.add_argument("--entrada", choices=["rostro", "senales"], default="rostro")
    ap.add_argument("--congelar", action="store_true", help="deja fija la CNN y entrena solo la LSTM")
    ap.add_argument("--rapido", action="store_true", help="prueba corta: pocas ventanas y 1 época")
    args = ap.parse_args()

    torch.manual_seed(config.SEMILLA); np.random.seed(config.SEMILLA)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre = (("m3" if args.particion == "sujeto" else "m3_aleatoria")
              + ("_senales" if args.entrada == "senales" else "") + ("_congelada" if args.congelar else "")
              + ("_rapido" if args.rapido else ""))

    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    d = d[d.fuente == "uta"]
    senales = args.entrada == "senales"
    if senales:  # antes de filtrar, para que el caché tenga todo UTA (M5 y el análisis de KSS 5)
        d = d.merge(senales_uta(d, disp), on="id")
    d = d[d.somnolencia.notna()]  # fuera KSS 5
    d = d[d.groupby("ventana").ventana.transform("size") >= MIN_FOTOGRAMAS]
    col = "particion" if args.particion == "sujeto" else "particion_aleatoria"
    partes = {p: d[d[col] == p] for p in ("entrenamiento", "validacion", "combinador", "prueba")}
    if args.particion == "sujeto":  # en la aleatoria no hay columna de balanceo; KSS 0 y 10 ya están parejos
        partes["entrenamiento"] = partes["entrenamiento"][partes["entrenamiento"].balanceado]
    if args.rapido:
        for p, x in partes.items():
            partes[p] = x[x.ventana.isin(x.ventana.drop_duplicates().sample(min(x.ventana.nunique(), 32), random_state=0))]

    hilos = 0 if senales else args.hilos  # las señales ya están en memoria
    opciones = dict(batch_size=args.lote, num_workers=hilos, pin_memory=True, persistent_workers=hilos > 0)
    datos = {p: (Senales if senales else Ventanas)(x, p == "entrenamiento") for p, x in partes.items()}
    c = {p: DataLoader(ds, shuffle=p == "entrenamiento", drop_last=p == "entrenamiento", **opciones)
         for p, ds in datos.items()}
    print(f"{nombre} en {disp}: " + ", ".join(f"{p} {len(ds)} ventanas" for p, ds in datos.items()))

    red = CnnLstm(args.congelar, senales).to(disp)
    if senales:  # estandarización con la media y la desviación del entrenamiento
        red.media.copy_(torch.tensor(partes["entrenamiento"][SENALES].mean().to_numpy(), dtype=torch.float32))
        red.desv.copy_(torch.tensor(partes["entrenamiento"][SENALES].std().to_numpy(), dtype=torch.float32))
    print(f"parámetros entrenables: {sum(p.numel() for p in red.parameters() if p.requires_grad):,}")
    perdida = nn.BCEWithLogitsLoss()  # el entrenamiento ya está balanceado por ventana
    mejor_f1 = entrenar(red, c["entrenamiento"], c["validacion"], perdida, nombre, 1 if args.rapido else EPOCAS, disp)

    # Prueba final con el mejor modelo (según F1 de validación)
    _, y_pru, p_pru = evaluar(red, c["prueba"], disp, perdida)
    resultado = {"modelo": nombre, "prueba": metricas(y_pru, p_pru),
                 "precision_promedio_AP": float(average_precision_score(y_pru, p_pru)),
                 **medir_latencia(red, disp),
                 "dispositivo": torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu",
                 "mejor_f1_validacion": mejor_f1}
    # Exactitud por sujeto de prueba: muestra si falla con personas puntuales
    v = partes["prueba"].groupby("ventana").sujeto.first().loc[datos["prueba"].ventanas]
    por_sujeto = pd.DataFrame({"sujeto": v.to_numpy(), "ok": (p_pru >= 0.5) == (y_pru == 1)}).groupby("sujeto").ok
    resultado["exactitud_por_sujeto"] = por_sujeto.mean().round(4).to_dict()
    (config.REPORTES / f"{nombre}_resultado.json").write_text(json.dumps(resultado, indent=2), encoding="utf8")

    # Salida z3 por ventana para entrenar y probar M5
    filas = []
    for p in ("combinador", "prueba"):
        _, y, prob = (None, y_pru, p_pru) if p == "prueba" else evaluar(red, c[p], disp, perdida)
        filas.append(pd.DataFrame({"ventana": datos[p].ventanas, "particion": p, "y": y, "z3": prob}))
    pd.concat(filas).to_csv(config.REPORTES / f"{nombre}_predicciones.csv", index=False)
    print(json.dumps(resultado, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
