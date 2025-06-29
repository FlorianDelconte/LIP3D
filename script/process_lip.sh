#!/bin/bash

# Début du chronomètre
start_time=$(date +%s)

# Chemin vers le répertoire racine
data_root="/volWork/these/DATA/ModelNet/lipCustom10"

# Script python à appeler
script_py="make_custom_feature_file.py"

# Fichiers de sortie pour train et test
train_output_file="${data_root}/EXP/2lipOm/train_caracs.txt"
test_output_file="${data_root}/EXP/2lipOm/test_caracs.txt"
train_names_file="${data_root}/EXP/2lipOm/train_names.txt"
test_names_file="${data_root}/EXP/2lipOm/test_names.txt"
train_labels_file="${data_root}/EXP/2lipOm/train_labels.txt"
test_labels_file="${data_root}/EXP/2lipOm/test_labels.txt"

# Vider ou créer les fichiers de sortie
> "$train_output_file"
> "$test_output_file"
> "$train_names_file"
> "$test_names_file"
> "$train_labels_file"
> "$test_labels_file"

# Initialisation du compteur pour les labels
label_counter=0

# Parcours des catégories
for categorie in "$data_root"/*; do
    # Vérifier si c'est bien un dossier
    if [ -d "$categorie" ]; then
        category_name=$(basename "$categorie")

        # Ignorer le répertoire EXP
        if [ "$category_name" == "EXP" ]; then
            echo "[Info] Catégorie ignorée : $category_name"
            continue
        fi

        category_label=$label_counter
        echo "[Info] Traitement de la catégorie : $category_name (label : $category_label)"
        ((label_counter++))

        for subset in "train" "test"; do
            subset_path="$categorie/$subset"

            carac_path="$subset_path/carac"
            pgm_path="$subset_path/pgm"

            # Déterminer les fichiers de sortie en fonction du sous-ensemble
            if [ "$subset" == "train" ]; then
                output_file="$train_output_file"
                names_file="$train_names_file"
                labels_file="$train_labels_file"
            else
                output_file="$test_output_file"
                names_file="$test_names_file"
                labels_file="$test_labels_file"
            fi

            echo "[Info] Début du traitement de : $subset_path"

            # Parcours des fichiers carac pour chaque objet
            for file_m in "$carac_path"/*_m.csv; do

                # Extraction du préfixe (nom de l'objet)
                filename=$(basename "$file_m" "_m.csv")
                echo "[Traitement] Objet : $filename"

                file_s="$carac_path/${filename}_s.csv"
                file_t="$carac_path/${filename}_t.csv"

                img_m="$pgm_path/${filename}_m.pgm"
                img_s="$pgm_path/${filename}_s.pgm"
                img_t="$pgm_path/${filename}_t.pgm"

                # Vérification de l'existence de tous les fichiers nécessaires
                if [[ -f "$file_s" && -f "$file_t" && -f "$img_m" && -f "$img_s" && -f "$img_t" ]]; then

                    # Appel du script python avec les arguments
                    python "$script_py" "$file_m" "$file_s" "$file_t" "$img_m" "$img_s" "$img_t" --use_orientation_merit >> "$output_file"

                    # Écriture du nom et du label
                    echo "$filename" >> "$names_file"
                    echo "$category_label" >> "$labels_file"
                else
                    echo "[Warning] Fichiers manquants pour l'objet : $filename dans $subset_path" >&2
                fi
            done

            echo "[Info] Traitement terminé pour : $subset_path"

        done
    fi
done

# Fin du chronomètre
end_time=$(date +%s)

# Calcul et affichage du temps d'exécution total
total_time=$((end_time - start_time))
echo "[Info] Temps total d'exécution : ${total_time} secondes"