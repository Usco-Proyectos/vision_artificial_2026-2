"""M1: ojo cerrado con MobileNetV3-Small sobre MRL Eye (ver docs/secciones/05-modelo-matematico.md).

- Etiqueta: y = 1 - ojo (1 = cerrado). Entrada: recorte gris 96x96 copiado a 3 canales y
  normalizado con la media y desviación de ImageNet.
- Pérdida: entropía cruzada binaria (el entrenamiento ya está balanceado, así que ω0 = ω1 = 1).
- Adam, η = 1e-3, lote 64, máx. 20 épocas; η se baja a la mitad si la pérdida de validación
  no mejora en 2 épocas; parada temprana si el F1 de validación no mejora en 4.
- Es reanudable: guarda un punto de control al final de cada época.

Uso (desde la carpeta src):
    python -m modelos.entrenar_m1                          # partición por sujeto (la real)
    python -m modelos.entrenar_m1 --particion aleatoria    # solo para medir la fuga
"""
import argparse
import json
import time

import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

import config

LOTE = 64
ETA = 1e-3
EPOCAS = 20
PACIENCIA_ETA = 2
PACIENCIA_PARADA = 4
UMBRAL = 0.5
MEDIA, DESV = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
DIR_MODELOS = config.RAIZ_DATOS / "modelos"


def a_tres_canales(t):
    """La red viene preentrenada con imágenes a color: el canal gris se copia 3 veces.
    Es una función normal (no lambda) para que Windows pueda pasarla a los procesos lectores."""
    return t.expand(3, -1, -1)


class Recortes(Dataset):
    """Recortes grises de 96x96 con su etiqueta. Lo usan M1 (ojos) y M2 (boca)."""

    def __init__(self, rutas, y, aumentar):
        self.rutas = [str(config.PROCESADO / r) for r in rutas]
        self.y = np.asarray(y, dtype=np.float32)
        pasos = [transforms.ToTensor()]  # gris uint8 -> 1x96x96 en [0, 1]
        if aumentar:
            pasos += [transforms.RandomHorizontalFlip(),
                      transforms.RandomRotation(10),
                      transforms.ColorJitter(brightness=0.3, contrast=0.3)]
        pasos += [transforms.Lambda(a_tres_canales),
                  transforms.Normalize(MEDIA, DESV)]
        self.tf = transforms.Compose(pasos)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        img = cv2.imread(self.rutas[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"No está el recorte {self.rutas[i]}. ¿Están los recortes en {config.RECORTES}?")
        return self.tf(img), self.y[i]


def crear_red():
    red = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    red.classifier[3] = nn.Linear(red.classifier[3].in_features, 1)  # una sola neurona
    return red


def metricas(y, p):
    pred = p >= UMBRAL
    vp = int(np.sum(pred & (y == 1))); vn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0))); fn = int(np.sum(~pred & (y == 1)))
    prec = vp / max(vp + fp, 1); sens = vp / max(vp + fn, 1)
    return {"exactitud": (vp + vn) / len(y), "precision": prec, "sensibilidad": sens,
            "f1": 2 * prec * sens / max(prec + sens, 1e-9), "VP": vp, "VN": vn, "FP": fp, "FN": fn,
            "n": len(y)}


@torch.no_grad()
def evaluar(red, cargador, disp, perdida):
    red.eval()
    ys, ps, total = [], [], 0.0
    for x, y in cargador:
        x, y = x.to(disp, non_blocking=True), y.to(disp)
        with torch.autocast(disp.type, enabled=disp.type == "cuda"):
            z = red(x).squeeze(1)
        total += perdida(z.float(), y).item() * len(y)
        ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(z.float()).cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return total / len(y), y, p


@torch.no_grad()
def medir_latencia(red, disp, repeticiones=300):
    red.eval()
    x = torch.randn(1, 3, config.TAM_OJO, config.TAM_OJO, device=disp)
    for _ in range(30):  # calentamiento, no se cuenta
        red(x)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(repeticiones):
        red(x)
    if disp.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / repeticiones * 1000
    return {"latencia_ms_por_imagen": ms, "imagenes_por_segundo": 1000 / ms}


def cargadores(ent, val, pru, col_ruta, y, hilos):
    """y(df) da la etiqueta 0/1 de cada fila. Solo el entrenamiento se baraja y se aumenta."""
    opciones = dict(batch_size=LOTE, num_workers=hilos, pin_memory=True, persistent_workers=hilos > 0)
    return (DataLoader(Recortes(ent[col_ruta], y(ent), True), shuffle=True, drop_last=True, **opciones),
            DataLoader(Recortes(val[col_ruta], y(val), False), shuffle=False, **opciones),
            DataLoader(Recortes(pru[col_ruta], y(pru), False), shuffle=False, **opciones))


def entrenar(red, c_ent, c_val, perdida, nombre, epocas, disp):
    """Bucle de entrenamiento común (Adam, η a la mitad, parada temprana por F1, reanudable).
    Deja el mejor modelo en <nombre>_mejor.pt y el historial en reportes; devuelve el mejor F1."""
    DIR_MODELOS.mkdir(parents=True, exist_ok=True)
    ruta_ctrl, ruta_mejor = DIR_MODELOS / f"{nombre}_control.pt", DIR_MODELOS / f"{nombre}_mejor.pt"
    ruta_hist = config.REPORTES / f"{nombre}_historial.csv"
    opt = torch.optim.Adam(red.parameters(), lr=ETA)
    planif = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=PACIENCIA_ETA)
    escalador = torch.amp.GradScaler(enabled=disp.type == "cuda")

    inicio, mejor_f1, sin_mejora = 0, -1.0, 0
    if ruta_ctrl.exists():  # reanudar
        c = torch.load(ruta_ctrl, map_location=disp, weights_only=False)
        red.load_state_dict(c["red"]); opt.load_state_dict(c["opt"]); planif.load_state_dict(c["planif"])
        escalador.load_state_dict(c["escalador"])
        inicio, mejor_f1, sin_mejora = c["epoca"] + 1, c["mejor_f1"], c["sin_mejora"]
        print(f"reanudando desde la época {inicio + 1}")

    for epoca in range(inicio, epocas):
        if sin_mejora >= PACIENCIA_PARADA:
            break
        red.train()
        t0, suma, n = time.time(), 0.0, 0
        for x, y in c_ent:
            x, y = x.to(disp, non_blocking=True), y.to(disp)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(disp.type, enabled=disp.type == "cuda"):
                z = red(x).squeeze(1)
            l = perdida(z.float(), y)
            escalador.scale(l).backward()
            escalador.step(opt); escalador.update()
            suma += l.item() * len(y); n += len(y)
        l_val, y_val, p_val = evaluar(red, c_val, disp, perdida)
        m_val = metricas(y_val, p_val)
        planif.step(l_val)
        if m_val["f1"] > mejor_f1:
            mejor_f1, sin_mejora = m_val["f1"], 0
            torch.save(red.state_dict(), ruta_mejor)
        else:
            sin_mejora += 1
        fila = {"epoca": epoca + 1, "perdida_ent": suma / n, "perdida_val": l_val,
                "exactitud_val": m_val["exactitud"], "f1_val": m_val["f1"],
                "eta": opt.param_groups[0]["lr"], "segundos": time.time() - t0}
        pd.DataFrame([fila]).to_csv(ruta_hist, mode="a", header=not ruta_hist.exists(), index=False)
        print(" | ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in fila.items()),
              flush=True)
        torch.save({"red": red.state_dict(), "opt": opt.state_dict(), "planif": planif.state_dict(),
                    "escalador": escalador.state_dict(), "epoca": epoca, "mejor_f1": mejor_f1,
                    "sin_mejora": sin_mejora}, ruta_ctrl)

    # Se devuelve la red con el mejor modelo (según F1 de validación), lista para la prueba final
    red.load_state_dict(torch.load(ruta_mejor, map_location=disp, weights_only=True))
    return mejor_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--particion", choices=["sujeto", "aleatoria"], default="sujeto")
    ap.add_argument("--hilos", type=int, default=2, help="procesos que leen imágenes")
    ap.add_argument("--rapido", action="store_true", help="prueba corta: pocas imágenes y 1 época")
    args = ap.parse_args()

    torch.manual_seed(config.SEMILLA); np.random.seed(config.SEMILLA)
    disp = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nombre = ("m1" if args.particion == "sujeto" else "m1_aleatoria") + ("_rapido" if args.rapido else "")
    epocas = 1 if args.rapido else EPOCAS

    d = pd.read_csv(config.PROCESADO / "dataset.csv", low_memory=False)
    d = d[d.fuente == "mrl"]
    col = "particion" if args.particion == "sujeto" else "particion_aleatoria"
    ent = d[(d[col] == "entrenamiento") & (d["balanceado"] if args.particion == "sujeto" else True)]
    val, pru = d[d[col] == "validacion"], d[d[col] == "prueba"]
    if args.rapido:
        ent, val, pru = (x.sample(min(len(x), 512), random_state=0) for x in (ent, val, pru))
    print(f"{nombre} en {disp}: entrenamiento {len(ent)}, validación {len(val)}, prueba {len(pru)}")

    c_ent, c_val, c_pru = cargadores(ent, val, pru, "ruta_ojo", lambda x: 1 - x["ojo"], args.hilos)  # 1 = cerrado
    red = crear_red().to(disp)
    print(f"parámetros: {sum(p.numel() for p in red.parameters()):,}")
    perdida = nn.BCEWithLogitsLoss()  # sigmoide + entropía cruzada binaria, en una sola operación estable
    mejor_f1 = entrenar(red, c_ent, c_val, perdida, nombre, epocas, disp)

    # Prueba final con el mejor modelo
    _, y_pru, p_pru = evaluar(red, c_pru, disp, perdida)
    resultado = {"modelo": nombre, "prueba": metricas(y_pru, p_pru), **medir_latencia(red, disp),
                 "dispositivo": torch.cuda.get_device_name(0) if disp.type == "cuda" else "cpu",
                 "mejor_f1_validacion": mejor_f1}
    # Desglose por condición (sirve para la discusión)
    pru = pru.assign(y=y_pru, p=p_pru)
    for cond in ["gafas", "iluminacion", "reflejo", "sensor"]:
        resultado[f"exactitud_por_{cond}"] = {
            str(k): round(float(((g.p >= UMBRAL) == (g.y == 1)).mean()), 4) for k, g in pru.groupby(cond)}
    (config.REPORTES / f"{nombre}_resultado.json").write_text(json.dumps(resultado, indent=2), encoding="utf8")
    pru[["id", "sujeto", "y", "p"]].to_csv(config.REPORTES / f"{nombre}_predicciones_prueba.csv", index=False)
    print(json.dumps(resultado, indent=2))


if __name__ == "__main__":
    main()
