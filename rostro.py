"""Conjuntos "mi rostro / otro rostro" y evaluacion, compartidos por las fases 2 y 3.

Separacion (siempre la misma, fija por la semilla):
  positivos  entrenamiento  sesion 1 menos 5 fotos   (aumentadas)
             validacion     5 fotos de la sesion 1   (aumentadas)
             prueba         sesiones 2 y 3           (sin aumentar, otro lugar y otra luz)
  negativos  caras de los fragmentos de CelebA reservados, que la red de la fase 1 nunca vio:
             1000 para prueba, 300 para validacion, el resto para entrenamiento.

La separacion de positivos es por foto original y por sesion ANTES de aumentar: si una foto
y sus variaciones cayeran en entrenamiento y prueba a la vez, la prueba saldria inflada.
"""
import os

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import comun

N_NEG_PRUEBA, N_NEG_VAL, N_POS_VAL = 1000, 300, 5

# Mismo aumento para positivos y negativos: si solo se aumentaran las fotos propias, la red
# podria aprender "imagen rotada o con bordes estirados = mi rostro" en vez de la cara.
_aumento = ImageDataGenerator(
    rotation_range=15, width_shift_range=0.08, height_shift_range=0.08, zoom_range=0.15,
    brightness_range=(0.6, 1.4), channel_shift_range=25, horizontal_flip=True, fill_mode="nearest",
)


def aumentar(imagenes, veces, semilla):
    """Devuelve `veces` variaciones aleatorias de cada imagen (uint8)."""
    x = np.repeat(np.asarray(imagenes, np.float32), veces, axis=0)
    salida = next(_aumento.flow(x, batch_size=len(x), shuffle=False, seed=semilla))
    return np.clip(salida, 0, 255).astype(np.uint8)


def conjuntos():
    rng = np.random.default_rng(comun.SEMILLA)
    propias = np.load(os.path.join(comun.DIR_DATOS, "propias_imagenes.npy"))
    sesion = np.load(os.path.join(comun.DIR_DATOS, "propias_sesion.npy"))
    celeba, _, fragmento, _ = comun.cargar_celeba()
    _, idx_reserva = comun.separar_reserva(fragmento)

    reserva = rng.permutation(idx_reserva)
    neg_prueba = np.asarray(celeba[np.sort(reserva[:N_NEG_PRUEBA])])
    neg_val = np.asarray(celeba[np.sort(reserva[N_NEG_PRUEBA:N_NEG_PRUEBA + N_NEG_VAL])])
    neg_ent = np.asarray(celeba[np.sort(reserva[N_NEG_PRUEBA + N_NEG_VAL:])])

    sesion1 = rng.permutation(np.flatnonzero(sesion == 1))
    pos_val, pos_ent = propias[sesion1[:N_POS_VAL]], propias[sesion1[N_POS_VAL:]]
    pos_prueba = propias[sesion >= 2]

    # Conjunto equilibrado: se aumentan las fotos propias hasta igualar el numero de negativos
    veces = int(np.ceil(len(neg_ent) / len(pos_ent)))
    x_pos = aumentar(pos_ent, veces, semilla=1)[:len(neg_ent)]
    x_neg = aumentar(neg_ent, 1, semilla=2)
    x_ent = np.concatenate([x_pos, x_neg])
    y_ent = np.concatenate([np.ones(len(x_pos)), np.zeros(len(x_neg))]).astype(np.float32)
    orden = rng.permutation(len(x_ent))

    veces_val = int(np.ceil(len(neg_val) / len(pos_val)))
    x_val = np.concatenate([aumentar(pos_val, veces_val, semilla=3)[:len(neg_val)], aumentar(neg_val, 1, semilla=4)])
    y_val = np.concatenate([np.ones(len(neg_val)), np.zeros(len(neg_val))]).astype(np.float32)

    x_prueba = np.concatenate([pos_prueba, neg_prueba])
    y_prueba = np.concatenate([np.ones(len(pos_prueba)), np.zeros(len(neg_prueba))]).astype(np.float32)
    tamanos = {
        "fotos_propias_originales": {"entrenamiento": len(pos_ent), "validacion": len(pos_val), "prueba": len(pos_prueba)},
        "entrenamiento": {"positivos_aumentados": len(x_pos), "negativos": len(x_neg)},
        "validacion": {"positivos_aumentados": len(neg_val), "negativos": len(neg_val)},
        "prueba": {"positivos": len(pos_prueba), "negativos": len(neg_prueba)},
    }
    return (x_ent[orden], y_ent[orden]), (x_val, y_val), (x_prueba, y_prueba), tamanos


def umbral_sin_falsos_positivos(prob_val, y_val):
    """Umbral elegido en VALIDACION: justo por encima de la cara ajena con mayor puntaje.

    En un desbloqueo, un falso positivo (abre con otra cara) es mucho peor que un falso
    negativo (hay que intentarlo otra vez). El umbral no se elige con la prueba para no
    ajustar el resultado a ella.
    """
    return float(np.nextafter(prob_val[y_val == 0].max(), 1.0))


def evaluar(prob, y, umbral_estricto):
    """Metricas en prueba al umbral 0.5 y al umbral estricto elegido en validacion."""
    def en_umbral(u):
        pred = prob >= u
        vp, fn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 1)))
        fp, vn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 0)))
        return {"umbral": round(u, 6), "VP": vp, "FN": fn, "FP": fp, "VN": vn,
                "exactitud": round((vp + vn) / len(y), 4),
                "precision": round(vp / (vp + fp), 4) if vp + fp else None,
                "exhaustividad": round(vp / (vp + fn), 4),
                "tasa_falsos_positivos": round(fp / (fp + vn), 4)}

    fpr, tpr, _ = roc_curve(y, prob)
    return {
        "auc": round(float(roc_auc_score(y, prob)), 4),
        "umbral_0.5": en_umbral(0.5),
        "umbral_estricto": en_umbral(umbral_estricto),
        "puntajes_positivos": [round(float(p), 4) for p in prob[y == 1]],
        "puntaje_maximo_negativo": round(float(prob[y == 0].max()), 4),
        "puntajes_negativos": [round(float(p), 6) for p in prob[y == 0]],
        "roc": {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()},
    }
