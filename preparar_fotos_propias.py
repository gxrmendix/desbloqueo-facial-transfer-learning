"""Recorta los rostros de las fotos propias con el mismo detector y encuadre que CelebA.

Estructura esperada (fuera del repositorio, nunca se versiona):
  ~/fotos-rostro/sesion1/*.jpg|heic|png   fotos propias, una carpeta por sesion
  ~/fotos-rostro/sesion2/...
  ~/fotos-rostro/otras-personas/...        opcional: caras de otras personas, con su permiso

Salida en datos/ (tambien ignorada por git):
  propias_imagenes.npy  (N, 64, 64, 3) uint8
  propias_sesion.npy    (N,) numero de sesion de cada foto
  otras_imagenes.npy    (M, 64, 64, 3) uint8, si existe la carpeta otras-personas
El resumen (cuantas fotos, cuantas sin rostro) va a resultados/preparacion_fotos_propias.json,
sin nombres de archivo.

Uso: python preparar_fotos_propias.py [--origen ~/fotos-rostro] [--hoja hoja.png]
"""
import argparse
import glob
import json
import os
import re
import subprocess
import tempfile

import cv2
import numpy as np

import recorte

EXTENSIONES = (".jpg", ".jpeg", ".png", ".heic", ".heif")


def leer_foto(ruta, tmp):
    """Lee una foto respetando la orientacion EXIF; las HEIC del iPhone se convierten con sips."""
    if ruta.lower().endswith((".heic", ".heif")):
        destino = os.path.join(tmp, os.path.basename(ruta) + ".jpg")
        subprocess.run(["sips", "-s", "format", "jpeg", ruta, "--out", destino], check=True, capture_output=True)
        ruta = destino
    return cv2.imread(ruta, cv2.IMREAD_COLOR)


def procesar_carpeta(carpeta, tmp):
    rutas = sorted(r for r in glob.glob(os.path.join(carpeta, "*")) if r.lower().endswith(EXTENSIONES))
    caras, sin_rostro = [], []
    for ruta in rutas:
        img = leer_foto(ruta, tmp)
        cara = recorte.recortar_rostro(img) if img is not None else None
        if cara is None:
            sin_rostro.append(os.path.basename(ruta))
        else:
            caras.append(cara)
    return caras, sin_rostro, len(rutas)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origen", default=os.path.expanduser("~/fotos-rostro"))
    ap.add_argument("--hoja", default=None, help="guarda una hoja de contacto de los recortes para revisarlos")
    args = ap.parse_args()

    sesiones = sorted(glob.glob(os.path.join(args.origen, "sesion*")),
                      key=lambda s: int(re.sub(r"\D", "", os.path.basename(s)) or 0))
    if not sesiones:
        raise SystemExit(f"no hay carpetas sesion*/ en {args.origen}")

    imagenes, sesion, resumen = [], [], {"sesiones": {}}
    with tempfile.TemporaryDirectory() as tmp:
        for numero, carpeta in enumerate(sesiones, start=1):
            caras, sin_rostro, total = procesar_carpeta(carpeta, tmp)
            imagenes += caras
            sesion += [numero] * len(caras)
            resumen["sesiones"][numero] = {"fotos": total, "con_rostro": len(caras), "sin_rostro": len(sin_rostro)}
            print(f"sesion {numero}: {total} fotos, {len(caras)} con rostro"
                  + (f", sin rostro detectado: {', '.join(sin_rostro)}" if sin_rostro else ""))
        otras_dir = os.path.join(args.origen, "otras-personas")
        otras = []
        if os.path.isdir(otras_dir):
            otras, sin_rostro, total = procesar_carpeta(otras_dir, tmp)
            resumen["otras_personas"] = {"fotos": total, "con_rostro": len(otras), "sin_rostro": len(sin_rostro)}
            print(f"otras personas: {total} fotos, {len(otras)} con rostro")

    np.save("datos/propias_imagenes.npy", np.array(imagenes, np.uint8))
    np.save("datos/propias_sesion.npy", np.array(sesion, np.int16))
    if otras:
        np.save("datos/otras_imagenes.npy", np.array(otras, np.uint8))
    os.makedirs("resultados", exist_ok=True)
    with open("resultados/preparacion_fotos_propias.json", "w") as f:
        json.dump(resumen, f, indent=1)

    if args.hoja:
        filas = [imagenes[i:i + 10] for i in range(0, len(imagenes), 10)]
        filas[-1] = filas[-1] + [np.zeros_like(imagenes[0])] * (10 - len(filas[-1]))
        hoja = np.vstack([np.hstack(f) for f in filas])
        cv2.imwrite(args.hoja, cv2.cvtColor(hoja, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
