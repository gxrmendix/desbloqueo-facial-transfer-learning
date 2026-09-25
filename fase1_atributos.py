"""Fase 1: red convolucional que predice los 40 atributos de CelebA.

Es un problema multi-etiqueta: cada atributo es independiente (una cara puede ser
"sonriente" y "con lentes" a la vez), asi que la salida son 40 sigmoides con
binary_crossentropy, no una softmax.

La red se divide en dos partes con nombre propio:
  - la base convolucional (bloques 1 a 4), que es lo que se reutiliza en la fase 2;
  - el clasificador denso, que se descarta despues.

Uso: python fase1_atributos.py [--epocas 15]
"""
import argparse
import json
import os
import time

import keras
import numpy as np
from keras import layers

import comun

TAMANO = 64


def bloque(x, filtros, n):
    for letra in "ab":
        x = layers.Conv2D(filtros, 3, padding="same", use_bias=False, name=f"bloque{n}_conv{letra}")(x)
        x = layers.BatchNormalization(name=f"bloque{n}_bn{letra}")(x)
        x = layers.ReLU(name=f"bloque{n}_relu{letra}")(x)
    return layers.MaxPooling2D(name=f"bloque{n}_pool")(x)


def construir_modelo(n_atributos):
    entrada = keras.Input((TAMANO, TAMANO, 3), dtype="uint8", name="imagen")
    x = layers.Rescaling(1 / 255, name="escala")(entrada)
    for n, filtros in enumerate([32, 64, 128, 256], start=1):
        x = bloque(x, filtros, n)
    # Promedio global en vez de Flatten: deja un vector de 256 rasgos que no depende de la
    # posicion exacta de la cara, util porque las selfies no quedan tan centradas como CelebA.
    x = layers.GlobalAveragePooling2D(name="rasgos")(x)
    x = layers.Dense(256, activation="relu", name="densa_atributos")(x)
    x = layers.Dropout(0.3, name="dropout_atributos")(x)
    salida = layers.Dense(n_atributos, activation="sigmoid", name="atributos")(x)
    return keras.Model(entrada, salida, name="red_atributos")


def lotes(imagenes, etiquetas, indices, tam_lote, voltear):
    """Generador que lee del arreglo en disco (mmap) solo el lote que hace falta."""
    rng = np.random.default_rng(comun.SEMILLA)
    while True:
        orden = rng.permutation(indices) if voltear else indices
        for i in range(0, len(orden), tam_lote):
            sel = np.sort(orden[i:i + tam_lote])
            x = np.asarray(imagenes[sel])
            if voltear:  # espejo horizontal al azar: una cara volteada tiene los mismos atributos
                espejo = rng.random(len(x)) < 0.5
                x[espejo] = x[espejo, :, ::-1]
            yield x, etiquetas[sel]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epocas", type=int, default=15)
    ap.add_argument("--lote", type=int, default=128)
    ap.add_argument("--max-pasos", type=int, default=None, help="solo para medir tiempos")
    args = ap.parse_args()
    keras.utils.set_random_seed(comun.SEMILLA)

    imagenes, atributos, fragmento, nombres = comun.cargar_celeba()
    idx_fase1, idx_reserva = comun.separar_reserva(fragmento)
    rng = np.random.default_rng(comun.SEMILLA)
    idx_fase1 = rng.permutation(idx_fase1)
    n_val = n_prueba = int(0.05 * len(idx_fase1))
    idx_val, idx_prueba = np.sort(idx_fase1[:n_val]), np.sort(idx_fase1[n_val:n_val + n_prueba])
    idx_ent = idx_fase1[n_val + n_prueba:]
    print(f"entrenamiento {len(idx_ent)}, validacion {len(idx_val)}, prueba {len(idx_prueba)}, "
          f"reserva para fase 2 {len(idx_reserva)}")

    modelo = construir_modelo(len(nombres))
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy",
                   metrics=[keras.metrics.BinaryAccuracy(name="exactitud")])
    modelo.summary(line_length=100)

    x_val, y_val = np.asarray(imagenes[idx_val]), atributos[idx_val]
    pasos = args.max_pasos or int(np.ceil(len(idx_ent) / args.lote))
    os.makedirs("modelos", exist_ok=True)
    inicio = time.time()
    historia = modelo.fit(
        lotes(imagenes, atributos, idx_ent, args.lote, voltear=True),
        steps_per_epoch=pasos, epochs=args.epocas, validation_data=(x_val, y_val), verbose=2,
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, verbose=1),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
            keras.callbacks.ModelCheckpoint("modelos/fase1_atributos.keras", monitor="val_loss", save_best_only=True),
        ],
    )
    minutos = (time.time() - inicio) / 60
    if args.max_pasos:
        print(f"{pasos} pasos en {minutos * 60:.1f} s")
        return

    # Evaluacion en prueba, atributo por atributo, contra la linea base de "siempre la clase mayoritaria"
    x_p, y_p = np.asarray(imagenes[idx_prueba]), atributos[idx_prueba]
    pred = (modelo.predict(x_p, batch_size=256, verbose=0) > 0.5).astype(np.uint8)
    exactitud = (pred == y_p).mean(0)
    frecuencia_ent = atributos[idx_ent].mean(0)
    base = np.where(frecuencia_ent > 0.5, y_p.mean(0), 1 - y_p.mean(0))
    por_atributo = {
        n: {"exactitud": round(float(e), 4), "linea_base": round(float(b), 4), "mejora": round(float(e - b), 4)}
        for n, e, b in zip(nombres, exactitud, base)
    }
    resumen = {
        "imagenes": {"entrenamiento": len(idx_ent), "validacion": len(idx_val), "prueba": len(idx_prueba),
                     "reserva_fase2": len(idx_reserva)},
        "parametros": int(modelo.count_params()),
        "epocas_corridas": len(historia.history["loss"]),
        "minutos_entrenamiento": round(minutos, 1),
        "exactitud_media_prueba": round(float(exactitud.mean()), 4),
        "linea_base_media_prueba": round(float(base.mean()), 4),
        "por_atributo": por_atributo,
        "historia": {k: [round(float(v), 5) for v in vs] for k, vs in historia.history.items()},
    }
    os.makedirs("resultados", exist_ok=True)
    with open("resultados/fase1_atributos.json", "w") as f:
        json.dump(resumen, f, indent=1, ensure_ascii=False)
    print(f"exactitud media en prueba {exactitud.mean():.4f} (linea base {base.mean():.4f}), {minutos:.1f} min")


if __name__ == "__main__":
    main()
