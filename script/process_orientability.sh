#!/bin/bash

# Liste des catégories à traiter
categories=("airplane" "guitar" "door" "lamp" "car")

# Dossier racine
root_dir="/volWork/these/DATA/ModelNet/lipMini"

# Script Python à appeler
script="orientability.py"

# Fichier de sortie CSV
output_csv="orientability.csv"
# Fichier des statistiques par catégorie
stats_csv="orientability_stats.csv"
# Nombre d'échantillons à traiter par catégorie
n=10

# En-têtes
echo "category,object_id,om_m,om_s,om_t" > "$output_csv"
echo "category,mean_om_m,std_om_m,mean_om_s,std_om_s,mean_om_t,std_om_t" > "$stats_csv"

for category in "${categories[@]}"; do
    echo "Processing $category..."

    carac_dir="$root_dir/$category/train/carac"
    files=($(ls "$carac_dir"/*_m.csv | sort | head -n $n))

    # Temporaire pour stocker les valeurs
    tmp_vals=()

    for m_file in "${files[@]}"; do
        base_name=$(basename "$m_file" _m.csv)
        s_file="$carac_dir/${base_name}_s.csv"
        t_file="$carac_dir/${base_name}_t.csv"

        if [[ -f "$s_file" && -f "$t_file" ]]; then
            output=$(python "$script" "$m_file" "$s_file" "$t_file")
            echo "$category,$base_name,$output" >> "$output_csv"
            tmp_vals+=("$output")
        else
            echo "Fichiers manquants pour $base_name dans $category"
        fi
    done

    # Si des résultats ont été trouvés
    if [[ ${#tmp_vals[@]} -gt 0 ]]; then
        # Extraire séparément les colonnes om_m, om_s, om_t
        om_m_list=()
        om_s_list=()
        om_t_list=()

        for line in "${tmp_vals[@]}"; do
            read -r om_m om_s om_t <<< "$line"
            om_m_list+=("$om_m")
            om_s_list+=("$om_s")
            om_t_list+=("$om_t")
        done

        # Fonction moyenne
        function mean() {
            awk '{sum+=$1} END {if (NR > 0) print sum/NR}'
        }

        # Fonction écart type
        function stddev() {
            awk '{
                sum+=$1; sumsq+=$1*$1
            } END {
                if (NR > 1) {
                    mean=sum/NR
                    std=sqrt(sumsq/NR - mean*mean)
                    print std
                } else {
                    print 0
                }
            }'
        }

        # Calcul stats
        mean_om_m=$(printf "%s\n" "${om_m_list[@]}" | mean)
        std_om_m=$(printf "%s\n" "${om_m_list[@]}" | stddev)

        mean_om_s=$(printf "%s\n" "${om_s_list[@]}" | mean)
        std_om_s=$(printf "%s\n" "${om_s_list[@]}" | stddev)

        mean_om_t=$(printf "%s\n" "${om_t_list[@]}" | mean)
        std_om_t=$(printf "%s\n" "${om_t_list[@]}" | stddev)

        # Arrondi à 6 chiffres
        printf "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n" "$category" "$mean_om_m" "$std_om_m" "$mean_om_s" "$std_om_s" "$mean_om_t" "$std_om_t" >> "$stats_csv"
    fi

    echo ""
done

echo "✅ Résultats enregistrés dans $output_csv"
