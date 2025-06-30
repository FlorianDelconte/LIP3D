import numpy as np
import pandas as pd
import argparse
import math
import sys

def orientation_merit(lip):
    """
    Calcule l'orientation merit à partir du vecteur LIP.
    """
    sdo = np.max(lip[:, 0])
    return 1 - math.exp(1 - sdo)

def main(argv):
    parser = argparse.ArgumentParser(description="Calcule uniquement l'orientation merit à partir des fichiers LIP.")
    parser.add_argument("feature_m", type=str, help="Chemin vers le fichier CSV de la direction m")
    parser.add_argument("feature_s", type=str, help="Chemin vers le fichier CSV de la direction s")
    parser.add_argument("feature_t", type=str, help="Chemin vers le fichier CSV de la direction t")
    args = parser.parse_args(argv)

    # Lecture des fichiers
    lip_m = pd.read_csv(args.feature_m, header=None).to_numpy()
    lip_s = pd.read_csv(args.feature_s, header=None).to_numpy()
    lip_t = pd.read_csv(args.feature_t, header=None).to_numpy()

    # Calcul des orientation merits
    om_m = orientation_merit(lip_m)
    om_s = orientation_merit(lip_s)
    om_t = orientation_merit(lip_t)

    # Arrondi et affichage
    om_values = [om_m, om_s, om_t]
    om_values = [round(v, 4) for v in om_values]
    print(*om_values)

if __name__ == "__main__":
    main(sys.argv[1:])
