#!/bin/bash

# Vérification du nombre d'arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_directory> <output_directory>"
    exit 1
fi

INPUT_DIR="$1"
OUTPUT_DIR="$2"

# Vérifier si le répertoire d'entrée existe
if [ ! -d "$INPUT_DIR" ]; then
    echo "Erreur : le répertoire d'entrée '$INPUT_DIR' n'existe pas."
    exit 1
fi

# Démarrage du chrono
START_TIME=$(date +%s)

# Parcours des sous-répertoires
for SUBDIR in "$INPUT_DIR"/*; do
    if [ -d "$SUBDIR" ]; then
        for PART in train test; do
            CURRENT_INPUT="$SUBDIR/$PART"
            if [ -d "$CURRENT_INPUT" ]; then
                # Chemin relatif à partir du dossier input
                REL_PATH="${SUBDIR#$INPUT_DIR/}"  # ex: "sous_rep"
                CURRENT_OUTPUT="$OUTPUT_DIR/$REL_PATH/$PART"

                # Sous-dossiers pour les sorties
                OUT_PGM="$CURRENT_OUTPUT/pgm/"
                OUT_CARAC="$CURRENT_OUTPUT/carac/"
                

                mkdir -p "$OUT_PGM" "$OUT_CARAC"

                # Parcourir tous les fichiers .off
                for MESH_FILE in "$CURRENT_INPUT"/*.off; do
                    if [ ! -f "$MESH_FILE" ]; then
                        echo "Aucun fichier .off trouvé dans '$CURRENT_INPUT'."
                        continue
                    fi

                    BASENAME=$(basename "$MESH_FILE" .off)

                    IMG1="$OUT_PGM/${BASENAME}_m.pgm"
                    IMG2="$OUT_PGM/${BASENAME}_s.pgm"
                    IMG3="$OUT_PGM/${BASENAME}_t.pgm"
                    IMG_pre="$OUT_PGM/${BASENAME}"

                    echo "Processing $MESH_FILE with imProfile..."
                    ./imProfile -i "$MESH_FILE" -o "$IMG_pre"

                    if [ ! -f "$IMG1" ] || [ ! -f "$IMG2" ] || [ ! -f "$IMG3" ]; then
                        echo "Erreur : échec de la génération des images pour $MESH_FILE"
                        continue
                    fi

                    OUT_IMG1="$OUT_CARAC/${BASENAME}_m_visu.png"
                    OUT_IMG2="$OUT_CARAC/${BASENAME}_s_visu.png"
                    OUT_IMG3="$OUT_CARAC/${BASENAME}_t_visu.png"

                    CSV1="$OUT_CARAC/${BASENAME}_m.csv"
                    CSV2="$OUT_CARAC/${BASENAME}_s.csv"
                    CSV3="$OUT_CARAC/${BASENAME}_t.csv"

                    echo "Processing images with lip_sign.py..."
                    python3 lip_sign.py "$IMG1" "$IMG2" "$IMG3" "$OUT_CARAC"

                    if [ ! -f "$OUT_IMG1" ] || [ ! -f "$OUT_IMG2" ] || [ ! -f "$OUT_IMG3" ] || \
                       [ ! -f "$CSV1" ] || [ ! -f "$CSV2" ] || [ ! -f "$CSV3" ]; then
                        echo "Erreur : échec de la génération des fichiers LIP pour $MESH_FILE"
                        continue
                    fi

                    echo "Traitement terminé pour $MESH_FILE"
                done
            fi
        done
    fi
done

# Fin du chrono
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))

MINUTES=$((DURATION / 60))
SECONDS=$((DURATION % 60))
echo "⏱️  Temps total d'exécution : ${MINUTES} min ${SECONDS} sec"

echo "✅ Traitement terminé pour tous les fichiers."
