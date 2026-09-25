"""Deteccion y recorte de rostros, comun a CelebA y a las fotos propias.

Las dos fuentes pasan por exactamente el mismo detector (Haar frontal de OpenCV) y
el mismo encuadre. Si CelebA se usara con su recorte original y las selfies con otro,
la red podria aprender a distinguir "encuadre de CelebA" contra "encuadre de selfie"
en lugar de "mi cara" contra "otra cara".
"""
import cv2
import numpy as np

TAMANO = 64          # lado de la imagen que recibe la red
MARGEN = 1.3         # el recuadro de Haar es muy justo; se amplia 30% para incluir frente y menton
LADO_DETECCION = 800 # las fotos del celular se reducen a este lado maximo solo para detectar

_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def detectar_rostro(img_bgr):
    """Devuelve (x, y, lado) del rostro mas grande en coordenadas de la imagen original, o None."""
    alto, ancho = img_bgr.shape[:2]
    escala = min(1.0, LADO_DETECCION / max(alto, ancho))
    chica = cv2.resize(img_bgr, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA) if escala < 1 else img_bgr
    gris = cv2.equalizeHist(cv2.cvtColor(chica, cv2.COLOR_BGR2GRAY))
    minimo = int(0.2 * min(gris.shape))
    caras = _detector.detectMultiScale(gris, scaleFactor=1.1, minNeighbors=5, minSize=(minimo, minimo))
    if len(caras) == 0:
        return None
    x, y, w, h = max(caras, key=lambda c: c[2] * c[3])
    return x / escala, y / escala, max(w, h) / escala


def recortar_rostro(img_bgr):
    """Recorta el rostro mas grande, lo lleva a TAMANO x TAMANO y lo devuelve en RGB uint8.

    Si el recuadro ampliado se sale de la foto, el borde se rellena replicando pixeles
    para no deformar la cara. Devuelve None si no se detecta ningun rostro.
    """
    caja = detectar_rostro(img_bgr)
    if caja is None:
        return None
    x, y, lado = caja
    cx, cy = x + lado / 2, y + lado / 2
    lado = lado * MARGEN
    x0, y0 = int(round(cx - lado / 2)), int(round(cy - lado / 2))
    x1, y1 = int(round(cx + lado / 2)), int(round(cy + lado / 2))
    alto, ancho = img_bgr.shape[:2]
    relleno = max(0, -x0, -y0, x1 - ancho, y1 - alto)
    if relleno:
        img_bgr = cv2.copyMakeBorder(img_bgr, relleno, relleno, relleno, relleno, cv2.BORDER_REPLICATE)
        x0, y0, x1, y1 = x0 + relleno, y0 + relleno, x1 + relleno, y1 + relleno
    cara = cv2.resize(img_bgr[y0:y1, x0:x1], (TAMANO, TAMANO), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(cara, cv2.COLOR_BGR2RGB)
