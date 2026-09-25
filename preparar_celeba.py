"""Convierte los fragmentos parquet de CelebA en arreglos listos para entrenar.

Entrada: datos/parquet/train-000NN-of-00132.parquet (descargados de
huggingface.co/datasets/huggan/CelebA-faces-with-attributes).
Salida en datos/:
  celeba_imagenes.npy   (N, 64, 64, 3) uint8, rostros recortados con recorte.py
  celeba_atributos.npy  (N, 40) uint8, cada atributo en 0/1
  celeba_fragmento.npy  (N,) numero de fragmento de origen, para separar conjuntos
  celeba_atributos.json nombres de los 40 atributos, en el orden de las columnas

Uso: python preparar_celeba.py
"""
import glob
import io
import json
import os
import re
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

import recorte

DIR_DATOS = "datos"
NO_ATRIBUTOS = {"image", "label", "image_id"}


def procesar_fragmento(ruta):
    cv2.setNumThreads(1)  # el paralelismo va por procesos, no dentro de OpenCV
    numero = int(re.search(r"train-(\d+)-of", ruta).group(1))
    tabla = pd.read_parquet(ruta)
    columnas = [c for c in tabla.columns if c not in NO_ATRIBUTOS]
    imagenes, atributos, descartadas = [], [], 0
    for _, fila in tabla.iterrows():
        datos = np.frombuffer(fila["image"]["bytes"], np.uint8)
        cara = recorte.recortar_rostro(cv2.imdecode(datos, cv2.IMREAD_COLOR))
        if cara is None:
            descartadas += 1
            continue
        imagenes.append(cara)
        # CelebA original usa -1/1; esta version puede traer 0/1 o booleanos. Se normaliza a 0/1.
        atributos.append([1 if float(fila[c]) > 0 else 0 for c in columnas])
    return numero, columnas, np.array(imagenes, np.uint8), np.array(atributos, np.uint8), len(tabla), descartadas


def es_parquet_valido(ruta):
    # 11 de los 132 fragmentos del repositorio de HuggingFace estan truncados de origen
    # (00011, 00020, 00028, 00031, 00059, 00085, 00086, 00088, 00118, 00120, 00129):
    # su hash coincide con el publicado pero les falta el pie "PAR1". Se omiten.
    with open(ruta, "rb") as f:
        f.seek(-4, os.SEEK_END)
        return f.read() == b"PAR1"


def main():
    rutas = sorted(glob.glob(os.path.join(DIR_DATOS, "parquet", "*.parquet")))
    danados = [r for r in rutas if not es_parquet_valido(r)]
    for r in danados:
        print("se omite fragmento danado:", r)
    rutas = [r for r in rutas if r not in danados]
    with Pool(max(1, os.cpu_count() // 2)) as pool:
        resultados = pool.map(procesar_fragmento, rutas)

    columnas = resultados[0][1]
    assert all(r[1] == columnas for r in resultados), "los fragmentos no tienen las mismas columnas"
    imagenes = np.concatenate([r[2] for r in resultados])
    atributos = np.concatenate([r[3] for r in resultados])
    fragmento = np.concatenate([np.full(len(r[2]), r[0], np.int16) for r in resultados])
    total = sum(r[4] for r in resultados)
    descartadas = sum(r[5] for r in resultados)

    np.save(os.path.join(DIR_DATOS, "celeba_imagenes.npy"), imagenes)
    np.save(os.path.join(DIR_DATOS, "celeba_atributos.npy"), atributos)
    np.save(os.path.join(DIR_DATOS, "celeba_fragmento.npy"), fragmento)
    with open(os.path.join(DIR_DATOS, "celeba_atributos.json"), "w") as f:
        json.dump(columnas, f, indent=1)

    resumen = {
        "fragmentos": len(rutas),
        "imagenes_leidas": int(total),
        "sin_rostro_detectado": int(descartadas),
        "porcentaje_descartado": round(100 * descartadas / total, 2),
        "imagenes_utiles": int(len(imagenes)),
        "atributos": len(columnas),
        "frecuencia_positiva_por_atributo": dict(zip(columnas, np.round(atributos.mean(0), 4).tolist())),
    }
    os.makedirs("resultados", exist_ok=True)
    with open("resultados/preparacion_celeba.json", "w") as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: v for k, v in resumen.items() if k != "frecuencia_positiva_por_atributo"}, indent=1))


if __name__ == "__main__":
    main()
