"""Junta mrl.csv, yawdd.csv y uta.csv en dataset.csv: limpieza, particiones y balanceo.

- Limpieza: quita duplicados exactos (mismo hash) y marca EAR/MAR atípicos. Para EAR se usa
  la regla de 3 RIC; para MAR no, porque los bostezos (la señal que buscamos) caen fuera de
  ese rango. En MAR solo se marcan valores imposibles (> MAR_MAX; un bostezo real llega a ~1,2).
- Partición por sujeto (secciones 8 del diseño):
    MRL y YawDD: 70 % entrenamiento, 15 % validación, 15 % prueba.
    UTA-RLDD:    50 % entrenamiento, 10 % validación, 20 % combinador, 20 % prueba.
- particion_aleatoria: mismas proporciones, pero repartiendo fotogramas (o ventanas,
  en UTA) al azar sin mirar el sujeto. Solo sirve para medir la fuga de información.
- balanceado: True en las filas de entrenamiento que quedan después de submuestrear
  la clase mayoritaria de cada etiqueta. Validación y prueba no se tocan.

Uso:  python -m dataset.consolidar   (desde la carpeta src)
"""
import json

import numpy as np
import pandas as pd

import config

PROPORCIONES = {
    "mrl": [("entrenamiento", .70), ("validacion", .15), ("prueba", .15)],
    "yawdd": [("entrenamiento", .70), ("validacion", .15), ("prueba", .15)],
    "uta": [("entrenamiento", .50), ("validacion", .10), ("combinador", .20), ("prueba", .20)],
}
ETIQUETA = {"mrl": "ojo", "yawdd": "bostezo", "uta": "somnolencia"}
MAR_MAX = 1.2


def repartir(tamanos, proporciones, rng):
    """Asigna una partición a cada unidad (sujeto, fotograma o ventana).

    `tamanos` es {unidad: número de registros}. Las unidades se barajan y cada una va a
    la partición a la que más registros le faltan para su proporción. Así las proporciones
    se cumplen en registros y no solo en número de sujetos (en MRL unos sujetos tienen
    muchas más imágenes que otros).
    """
    unidades = sorted(tamanos)
    rng.shuffle(unidades)
    total = sum(tamanos.values())
    llevan = {nombre: 0 for nombre, _ in proporciones}
    asignacion = {}
    for u in unidades:
        nombre = max(proporciones, key=lambda np_: np_[1] * total - llevan[np_[0]])[0]
        asignacion[u] = nombre
        llevan[nombre] += tamanos[u]
    return asignacion


def repartir_sujetos(df, etiqueta, proporciones, rng, intentos=500, min_sujetos=4):
    """Prueba varios repartos por sujeto y se queda con el más parejo.

    Puntaje = desvío de las proporciones en registros + desvío de la proporción de la
    etiqueta en cada partición respecto a la global, con castigo si una partición queda
    con menos de `min_sujetos` sujetos. Hace falta en MRL, donde un sujeto tiene 10 257
    imágenes y otro 382, y algunos son casi todos ojo abierto.
    """
    tamanos = df["sujeto"].value_counts().to_dict()
    con_etq = df[df[etiqueta].notna()]
    pos = con_etq.groupby("sujeto")[etiqueta].sum().to_dict()
    n_etq = con_etq["sujeto"].value_counts().to_dict()
    total, global_pos = sum(tamanos.values()), con_etq[etiqueta].mean()
    mejor, mejor_puntaje = None, np.inf
    for _ in range(intentos):
        asig = repartir(tamanos, proporciones, rng)
        puntaje = 0.0
        for nombre, p in proporciones:
            miembros = [s for s, v in asig.items() if v == nombre]
            puntaje += abs(sum(tamanos[s] for s in miembros) / total - p)
            n = sum(n_etq.get(s, 0) for s in miembros)
            puntaje += abs(sum(pos.get(s, 0) for s in miembros) / n - global_pos) if n else 1
            puntaje += len(miembros) < min_sujetos
        if puntaje < mejor_puntaje:
            mejor, mejor_puntaje = asig, puntaje
    return mejor


def balancear(df, etiqueta, rng, unidad=None):
    """Submuestrea la clase mayoritaria en entrenamiento (por ventana si se da `unidad`)."""
    ent = df[(df["particion"] == "entrenamiento") & df[etiqueta].notna()]
    clave = unidad or "id"
    grupos = ent.groupby(clave)[etiqueta].first()
    minimo = grupos.value_counts().min()
    elegidos = set()
    for _, g in grupos.groupby(grupos):
        elegidos |= set(rng.choice(g.index.to_numpy(), size=minimo, replace=False))
    return df[clave].isin(elegidos) & (df["particion"] == "entrenamiento")


def main():
    rng = np.random.default_rng(config.SEMILLA)
    partes, reporte, particiones = [], {}, {}
    for fuente in ["mrl", "yawdd", "uta"]:
        ruta = config.PROCESADO / f"{fuente}.csv"
        if not ruta.exists():
            print(f"{fuente}: no hay {ruta.name}, se omite")
            continue
        df = pd.read_csv(ruta)
        n0 = len(df)
        df = df.drop_duplicates(subset="hash")
        dup = n0 - len(df)

        # EAR/MAR atípicos (fallas de MediaPipe): se marcan, no se borran
        df["atipico"] = False
        if "mar" in df:
            df["atipico"] |= df["mar"] > MAR_MAX
        for col in ["ear_der", "ear_izq"]:
            if col in df:
                q1, q3 = df[col].quantile([.25, .75])
                rango = q3 - q1
                df["atipico"] |= (df[col] < q1 - 3 * rango) | (df[col] > q3 + 3 * rango)

        etq = ETIQUETA[fuente]
        sujetos = repartir_sujetos(df, etq, PROPORCIONES[fuente], rng)
        df["particion"] = df["sujeto"].map(sujetos)
        unidad = "ventana" if fuente == "uta" else "id"
        aleatoria = repartir(df[unidad].value_counts().to_dict(), PROPORCIONES[fuente], rng)
        df["particion_aleatoria"] = df[unidad].map(aleatoria)

        df["balanceado"] = balancear(df, etq, rng, unidad="ventana" if fuente == "uta" else None)

        particiones[fuente] = {p: sorted(s for s, v in sujetos.items() if v == p)
                               for p, _ in PROPORCIONES[fuente]}
        reporte[fuente] = {
            "registros_entrada": n0, "duplicados_quitados": dup, "registros": len(df),
            "sujetos": int(df["sujeto"].nunique()), "atipicos_marcados": int(df["atipico"].sum()),
            "por_particion": df.groupby("particion")[etq].value_counts().unstack(fill_value=0).to_dict("index"),
            "entrenamiento_balanceado": df[df["balanceado"]][etq].value_counts().to_dict(),
        }
        partes.append(df)

    if not partes:
        raise SystemExit("No hay nada que consolidar todavía")
    total = pd.concat(partes, ignore_index=True)
    total.to_csv(config.PROCESADO / "dataset.csv", index=False)
    (config.REPORTES / "particiones.json").write_text(json.dumps(particiones, indent=2), encoding="utf8")
    (config.REPORTES / "resumen.json").write_text(json.dumps(reporte, indent=2, default=str), encoding="utf8")
    print(f"dataset.csv: {len(total)} registros, {total['sujeto'].nunique()} sujetos")
    print(json.dumps(reporte, indent=2, default=str))


if __name__ == "__main__":
    main()
