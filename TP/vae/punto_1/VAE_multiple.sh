#!/bin/bash

output_file="metrics.csv"
echo "run,nelbo,kl,rec" > $output_file

for i in $(seq 0 9); do
    echo "Ejecutando run $i"
    # Ejecutar y capturar la salida en una variable
    output=$(python ../run_vae.py --run $i)

    # Extraer la última línea que contenga NELBO
    line=$(echo "$output" | grep "NELBO:" | tail -1)

    # Extraer los valores con regex
    nelbo=$(echo "$line" | grep -oP 'NELBO: \K[\d\.e+-]+')
    kl=$(echo "$line" | grep -oP 'KL: \K[\d\.e+-]+')
    rec=$(echo "$line" | grep -oP 'Rec: \K[\d\.e+-]+')

    echo "$i,$nelbo,$kl,$rec" >> $output_file
done

echo "Resultados guardados en $output_file"

