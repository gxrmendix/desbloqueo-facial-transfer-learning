#!/bin/sh
# Descarga los primeros N fragmentos sanos de CelebA con sus 40 atributos (sin credenciales).
# Cada fragmento tiene 1535 imagenes, ~10.5 MB. Con N=30 son ~46 000 imagenes y ~315 MB.
# Algunos fragmentos del repositorio estan truncados de origen (sin el pie "PAR1");
# se detectan leyendo solo sus ultimos 4 bytes y se saltan.
N=${1:-30}
URL=https://huggingface.co/datasets/huggan/CelebA-faces-with-attributes/resolve/main/data
mkdir -p datos/parquet
i=0; bajados=0
while [ $bajados -lt $N ] && [ $i -lt 132 ]; do
  f=$(printf "train-%05d-of-00132.parquet" $i)
  if [ "$(curl -sfL -r -4 "$URL/$f" | xxd -p)" = "50415231" ]; then
    [ -f datos/parquet/$f ] || curl -sfL --retry 3 -o datos/parquet/$f "$URL/$f"
    bajados=$((bajados + 1))
  else
    echo "se salta $f (danado en el origen)"
  fi
  i=$((i + 1))
done
echo "$bajados fragmentos en datos/parquet/"
