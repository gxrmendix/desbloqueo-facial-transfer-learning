"""Figuras del README a partir de los resultados guardados en resultados/*.json.

No usa ninguna foto propia: solo metricas y puntajes.
Uso: python figuras.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SUPERFICIE, TINTA, TINTA_2, TENUE, REJILLA, EJE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE, "savefig.facecolor": SUPERFICIE,
    "axes.edgecolor": EJE, "axes.labelcolor": TINTA_2, "xtick.color": TENUE, "ytick.color": TENUE,
    "text.color": TINTA, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.6, "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
})


def leer(nombre):
    ruta = os.path.join("resultados", nombre)
    return json.load(open(ruta)) if os.path.exists(ruta) else None


def guardar(fig, nombre):
    os.makedirs("figuras", exist_ok=True)
    fig.savefig(os.path.join("figuras", nombre), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("figuras/" + nombre)


def fase1_perdida(r):
    h = r["historia"]
    epocas = np.arange(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for clave, color, texto in [("loss", AZUL, "entrenamiento"), ("val_loss", NARANJA, "validación")]:
        ax.plot(epocas, h[clave], color=color, lw=2, marker="o", ms=4, label=texto)
    mejor = int(np.argmin(h["val_loss"])) + 1
    ax.axvline(mejor, color=EJE, lw=1, ls="--")
    ax.annotate(f"mejor época ({mejor}), la que se guarda", (mejor, max(h["val_loss"])), xytext=(-6, 0),
                textcoords="offset points", ha="right", color=TINTA_2, fontsize=9)
    ax.set(xlabel="época", ylabel="pérdida (binary cross-entropy)",
           title="Fase 1 · pérdida de la red de 40 atributos")
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(1, 0.85))
    guardar(fig, "fase1_perdida.png")


def fase1_atributos(r):
    pa = sorted(r["por_atributo"].items(), key=lambda kv: kv[1]["mejora"])
    nombres = [k.replace("_", " ") for k, _ in pa]
    red = np.array([v["exactitud"] for _, v in pa]) * 100
    base = np.array([v["linea_base"] for _, v in pa]) * 100
    y = np.arange(len(pa))
    fig, ax = plt.subplots(figsize=(7.5, 10))
    ax.hlines(y, base, red, color=EJE, lw=2)
    ax.scatter(base, y, color=TENUE, s=36, zorder=3, label="línea base: siempre la clase mayoritaria",
               edgecolor=SUPERFICIE, linewidth=1.5)
    ax.scatter(red, y, color=AZUL, s=40, zorder=4, label="red de la fase 1", edgecolor=SUPERFICIE, linewidth=1.5)
    ax.set_yticks(y, nombres, fontsize=8.5)
    ax.set_xlim(45, 101)
    ax.set_xlabel("exactitud en prueba (%)")
    ax.set_title("Fase 1 · exactitud por atributo contra la línea base\n(ordenados por cuánto mejora la red)", pad=26)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.grid(axis="y", visible=False)
    guardar(fig, "fase1_atributos.png")


def curva_desbloqueo(f2, f3):
    """Cuantas caras ajenas desbloquean (de 1000) contra cuantas fotos propias se reconocen."""
    series = [(f2["control_pesos_al_azar"], NARANJA, "control: base al azar"),
              (f2["preentrenada_celeba"], AZUL, "fase 2: base CelebA congelada")]
    if f3:
        series.append((f3, AQUA, "fase 3: último bloque refinado"))
    n_pos = f2["conjuntos"]["prueba"]["positivos"]
    n_neg = f2["conjuntos"]["prueba"]["negativos"]
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for r, color, texto in series:
        roc = r["prueba"]["roc"]
        x, y = np.array(roc["fpr"]) * n_neg, np.array(roc["tpr"]) * n_pos
        ax.step(x, y, where="post", color=color, lw=2, label=f"{texto} (AUC {r['prueba']['auc']:.4f})")
    ax.set_xlim(-1, 60)
    ax.set_ylim(0, n_pos + 0.5)
    ax.set_yticks(range(0, n_pos + 1))
    ax.set_xlabel(f"caras ajenas que desbloquean (de {n_neg})")
    ax.set_ylabel(f"fotos propias reconocidas (de {n_pos})")
    ax.set_title("¿Cuántos extraños hay que dejar pasar para reconocerme?\n(cada curva recorre todos los umbrales posibles)")
    ax.legend(frameon=False, loc="lower right")
    guardar(fig, "curva_desbloqueo.png")


def puntajes(r, titulo, nombre):
    """Puntajes en escala logit: casi todos estan pegados a 0 o a 1 y en escala lineal no se distinguen."""
    p = r["prueba"]
    piso = 1e-6
    logit = lambda q: np.log(np.clip(q, piso, 1 - piso) / (1 - np.clip(q, piso, 1 - piso)))
    propios = logit(np.array(p["puntajes_positivos"]))
    ajenos = logit(np.array(p["puntajes_negativos"]))
    tope = 60  # la barra del piso tiene cientos de caras; se recorta para que se vea la zona de los umbrales
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    conteos, bordes, _ = ax.hist(ajenos, bins=np.linspace(logit(piso), logit(1 - piso), 61), color=TENUE,
                                 label=f"caras ajenas ({len(ajenos)})")
    if conteos.max() > tope:
        i = int(np.argmax(conteos))
        ax.annotate(f"{int(conteos[i])} caras con puntaje casi 0 ↑", ((bordes[i] + bordes[i + 1]) / 2, tope),
                    xytext=(8, -12), textcoords="offset points", color=TINTA_2, fontsize=9)
    ax.set_ylim(-12, tope)
    ax.scatter(propios, np.full(len(propios), -6), marker="|", s=250, color=AZUL, lw=2.5,
               label=f"mis fotos de prueba ({len(propios)})", zorder=4)
    for u, texto, ha, dx in [(0.5, "umbral 0.5", "right", -4), (p["umbral_estricto"]["umbral"], "umbral estricto", "left", 4)]:
        ax.axvline(logit(u), color=TINTA_2, lw=1, ls="--")
        ax.annotate(texto, (logit(u), tope), xytext=(dx, -30), textcoords="offset points",
                    color=TINTA_2, fontsize=9, ha=ha)
    ticks = [piso, 0.001, 0.1, 0.5, 0.9, 0.999, 1 - piso]
    etiquetas = ["≤0.000001", "0.001", "0.1", "0.5", "0.9", "0.999", "≥0.999999"]
    ax.set_xticks(logit(np.array(ticks)), etiquetas)
    ax.set_xlabel("puntaje de la red: probabilidad de que sea mi rostro (escala ampliada en los extremos)")
    ax.set_ylabel("número de caras")
    ax.set_title(titulo)
    ax.set_yticks(range(0, tope + 1, 10))
    ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.05, 0.66))
    guardar(fig, nombre)


def main():
    f1, f2, f3 = leer("fase1_atributos.json"), leer("fase2_clasificador.json"), leer("fase3_refinamiento.json")
    if f1:
        fase1_perdida(f1)
        fase1_atributos(f1)
    if f2:
        curva_desbloqueo(f2, f3)
        final = f3 or f2["preentrenada_celeba"]
        puntajes(final, "Puntajes en el conjunto de prueba · modelo final", "puntajes_prueba.png")


if __name__ == "__main__":
    main()
