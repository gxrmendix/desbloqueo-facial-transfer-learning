"""Fase 2: base convolucional de la fase 1 congelada + clasificador "es mi rostro o no".

Pasos del enunciado:
  2. se construye un modelo con las capas convolucionales de la red de atributos,
     quitandole el clasificador denso;
  3. se agrega un clasificador de una capa densa y una sola neurona de salida;
  4. se congelan los pesos de la parte pre-entrenada;
  5. se entrena.

Como control se entrena el mismo clasificador sobre la misma arquitectura pero con pesos
convolucionales al azar (tambien congelados). La diferencia entre ambos es lo que aporta
el pre-entrenamiento con CelebA.

Uso: python fase2_clasificador.py
"""
import json
import os

import keras
import numpy as np
from keras import layers

import comun
import rostro
from fase1_atributos import construir_modelo


def base_convolucional(red_atributos):
    """Todas las capas hasta el ultimo bloque convolucional; se descarta el clasificador denso."""
    return keras.Model(red_atributos.input, red_atributos.get_layer("bloque4_pool").output, name="base_convolucional")


def modelo_rostro(base):
    base.trainable = False
    entrada = keras.Input((64, 64, 3), dtype="uint8", name="imagen")
    x = base(entrada, training=False)  # BatchNormalization en modo inferencia, con las estadisticas de CelebA
    x = layers.GlobalAveragePooling2D(name="rasgos")(x)
    x = layers.Dense(128, activation="relu", name="densa_rostro")(x)
    x = layers.Dropout(0.5, name="dropout_rostro")(x)
    salida = layers.Dense(1, activation="sigmoid", name="es_mi_rostro")(x)
    return keras.Model(entrada, salida, name="desbloqueo_facial")


def entrenar(modelo, ent, val, tasa, epocas, ruta=None):
    modelo.compile(optimizer=keras.optimizers.Adam(tasa), loss="binary_crossentropy",
                   metrics=[keras.metrics.BinaryAccuracy(name="exactitud"), keras.metrics.AUC(name="auc")])
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1)]
    if ruta:
        callbacks.append(keras.callbacks.ModelCheckpoint(ruta, monitor="val_loss", save_best_only=True))
    h = modelo.fit(*ent, validation_data=val, epochs=epocas, batch_size=64, verbose=2, callbacks=callbacks)
    return {k: [round(float(v), 5) for v in vs] for k, vs in h.history.items()}


def main():
    keras.utils.set_random_seed(comun.SEMILLA)
    ent, val, prueba, tamanos = rostro.conjuntos()
    print(json.dumps(tamanos))

    resultados = {"conjuntos": tamanos}
    red_atributos = keras.models.load_model("modelos/fase1_atributos.keras")
    variantes = {
        "preentrenada_celeba": base_convolucional(red_atributos),
        "control_pesos_al_azar": base_convolucional(construir_modelo(40)),
    }
    for nombre, base in variantes.items():
        print(f"\n=== {nombre} ===")
        modelo = modelo_rostro(base)
        if nombre == "preentrenada_celeba":
            modelo.summary(line_length=100)
        ruta = "modelos/fase2_rostro.keras" if nombre == "preentrenada_celeba" else None
        # Limite alto para que el control con pesos al azar tambien converja: con 40 epocas
        # seguia mejorando y la comparacion habria favorecido a la base pre-entrenada.
        # La parada temprana corta cada variante cuando deja de mejorar.
        historia = entrenar(modelo, ent, val, tasa=1e-3, epocas=150, ruta=ruta)
        prob_val = modelo.predict(val[0], verbose=0).ravel()
        umbral = rostro.umbral_sin_falsos_positivos(prob_val, val[1])
        prob = modelo.predict(prueba[0], verbose=0).ravel()
        resultados[nombre] = {
            "parametros_entrenables": int(sum(np.prod(w.shape) for w in modelo.trainable_weights)),
            "parametros_congelados": int(sum(np.prod(w.shape) for w in modelo.non_trainable_weights)),
            "epocas": len(historia["loss"]), "historia": historia,
            "prueba": rostro.evaluar(prob, prueba[1], umbral),
        }
        r = resultados[nombre]["prueba"]
        print(f"{nombre}: AUC {r['auc']} | umbral 0.5 -> {r['umbral_0.5']} | estricto -> {r['umbral_estricto']}")

    os.makedirs("resultados", exist_ok=True)
    with open("resultados/fase2_clasificador.json", "w") as f:
        json.dump(resultados, f, indent=1)


if __name__ == "__main__":
    main()
