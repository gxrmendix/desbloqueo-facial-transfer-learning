"""Fase 3: refinamiento del ultimo bloque convolucional junto con el clasificador.

Se parte del modelo de la fase 2 (clasificador ya entrenado sobre la base congelada) y se
descongelan las dos convoluciones del bloque 4. Las capas de BatchNormalization siguen
congeladas: con lotes pequenos y tan pocas fotos propias, recalcular sus estadisticas
desajustaria lo aprendido con CelebA. La tasa de aprendizaje es 100 veces menor que en la
fase 2 para mover los pesos pre-entrenados solo un poco.

Uso: python fase3_refinamiento.py
"""
import json

import keras
import numpy as np

import comun
import rostro
from fase2_clasificador import entrenar

CAPAS_A_REFINAR = ("bloque4_conva", "bloque4_convb")


def main():
    keras.utils.set_random_seed(comun.SEMILLA)
    ent, val, prueba, tamanos = rostro.conjuntos()

    modelo = keras.models.load_model("modelos/fase2_rostro.keras")
    base = modelo.get_layer("base_convolucional")
    base.trainable = True
    for capa in base.layers:
        capa.trainable = capa.name in CAPAS_A_REFINAR
    entrenables = [c.name for c in base.layers if c.trainable]
    print("capas convolucionales que se refinan:", entrenables)

    historia = entrenar(modelo, ent, val, tasa=1e-5, epocas=30, ruta="modelos/fase3_rostro.keras")
    prob_val = modelo.predict(val[0], verbose=0).ravel()
    umbral = rostro.umbral_sin_falsos_positivos(prob_val, val[1])
    prob = modelo.predict(prueba[0], verbose=0).ravel()
    resultados = {
        "capas_refinadas": entrenables,
        "parametros_entrenables": int(sum(np.prod(w.shape) for w in modelo.trainable_weights)),
        "epocas": len(historia["loss"]), "historia": historia,
        "prueba": rostro.evaluar(prob, prueba[1], umbral),
    }
    r = resultados["prueba"]
    print(f"fase 3: AUC {r['auc']} | umbral 0.5 -> {r['umbral_0.5']} | estricto -> {r['umbral_estricto']}")
    with open("resultados/fase3_refinamiento.json", "w") as f:
        json.dump(resultados, f, indent=1)


if __name__ == "__main__":
    main()
