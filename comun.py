"""Carga de datos y separacion de conjuntos compartidas por las tres fases."""
import json
import os

import numpy as np

DIR_DATOS = "datos"
SEMILLA = 42
N_FRAGMENTOS_RESERVADOS = 3  # los ultimos 3 fragmentos no se usan en la fase 1


def cargar_celeba():
    imagenes = np.load(os.path.join(DIR_DATOS, "celeba_imagenes.npy"), mmap_mode="r")
    atributos = np.load(os.path.join(DIR_DATOS, "celeba_atributos.npy"))
    fragmento = np.load(os.path.join(DIR_DATOS, "celeba_fragmento.npy"))
    with open(os.path.join(DIR_DATOS, "celeba_atributos.json")) as f:
        nombres = json.load(f)
    return imagenes, atributos, fragmento, nombres


def separar_reserva(fragmento):
    """Indices de la fase 1 y de la reserva (caras que la red base nunca ve).

    La reserva sale de fragmentos completos, no de un muestreo al azar, para que ni
    siquiera fotos vecinas del mismo lote de CelebA pasen por el entrenamiento de la fase 1.
    """
    reservados = np.unique(fragmento)[-N_FRAGMENTOS_RESERVADOS:]
    en_reserva = np.isin(fragmento, reservados)
    return np.flatnonzero(~en_reserva), np.flatnonzero(en_reserva)
