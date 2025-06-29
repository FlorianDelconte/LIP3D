from skimage.transform import radon, rescale
from skimage.measure import label, regionprops, perimeter_crofton
from skimage.morphology import area_closing
from skimage import io
from scipy.signal import find_peaks
import numpy as np
import scipy.fft
import pandas as pd
import cv2
import sys
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
import argparse

def local_features(lip, stats=("max", "min", "median", "mean", "std"), mode="default"):
    """
    Extrait des caractéristiques locales à partir d'une matrice LIP.

    Parameters:
        lip (ndarray): Matrice de signatures LIP (chaque colonne est une signature).
        stats (tuple): Statistiques à extraire ("max", "min", "median", "mean", "std").
        mode (str): Mode d'extraction. 
                    - "default" : statistiques classiques sur chaque colonne.
                    - "ref_by_LIP0" : min, max, median extraits selon les indices de LIP0.

    Returns:
        ndarray: Vecteur de caractéristiques locales.
    """
    local_feature = []

    if mode == "default":
        for i in range(lip.shape[1]):
            signature = lip[:, i]

            if "max" in stats:
                local_feature.append(np.max(signature))
            if "min" in stats:
                local_feature.append(np.min(signature))
            if "median" in stats:
                local_feature.append(np.median(signature))
            if "mean" in stats:
                local_feature.append(np.mean(signature))
            if "std" in stats:
                local_feature.append(np.std(signature))

    elif mode == "ref_by_LIP0":
        lip0 = lip[:, 0]
        idx_min = np.argmin(lip0)
        idx_max = np.argmax(lip0)
        idx_med = np.argsort(lip0)[len(lip0) // 2]

        for i in range(lip.shape[1]):
            signature = lip[:, i]

            if "min" in stats:
                local_feature.append(signature[idx_min])
            if "max" in stats:
                local_feature.append(signature[idx_max])
            if "median" in stats:
                local_feature.append(signature[idx_med])
            # Les stats "mean" et "std" restent globales
            if "mean" in stats:
                local_feature.append(np.mean(signature))
            if "std" in stats:
                local_feature.append(np.std(signature))
    elif mode == "ref_by_LIP0_fft":
        lip0 = lip[:, 0]
        representative_indices, _, _, _ = select_lip_representative_indices(
            lip0, top_n_freq=10, n_representatives=10
        )

        for i in range(lip.shape[1]):
            signature = lip[:, i]
            for idx in representative_indices:
                local_feature.append(signature[idx])
            
    else:
        raise ValueError(f"Mode '{mode}' non reconnu. Utilisez 'default' ou 'ref_by_LIP0' ou 'ref_by_LIP0_fft'.")

    return np.array(local_feature)

def select_lip_representative_indices(lip0, top_n_freq=10, n_representatives=10):
    """
    Sélectionne un nombre fixe de représentants à partir de la reconstruction FFT de lip0,
    en répartissant de façon équilibrée les pics, creux et plateaux.
    Complète si nécessaire avec les points les plus extrêmes.
    """
    top_indices = lip_fft_representatives(lip0, top_n=top_n_freq)
    reconstructed = reconstruct_signal_from_fft(lip0, top_indices)
    features = detect_signal_features(reconstructed, distance=10)

    # Nombre cible par catégorie
    per_type_target = n_representatives // 3
    remaining = n_representatives - 3 * per_type_target

    selected = []

    # Prendre des pics
    peaks = features['peaks'][:per_type_target]
    selected.extend(peaks)

    # Prendre des creux
    troughs = features['troughs'][:per_type_target]
    selected.extend(troughs)

    # Prendre des plateaux
    plateaux = features['plateau_indices'][:per_type_target + remaining]
    selected.extend(plateaux)

    # S'assurer qu'ils sont uniques et compléter si nécessaire
    seen = set(selected)
    if len(seen) < n_representatives:
        fallback = np.argsort(-np.abs(reconstructed))
        for idx in fallback:
            if idx not in seen:
                selected.append(idx)
                seen.add(idx)
            if len(seen) >= n_representatives:
                break

    return sorted(selected[:n_representatives]), top_indices, reconstructed, features

def detect_signal_features(signal, distance=10):
    peaks, _ = find_peaks(signal, distance=distance)
    troughs, _ = find_peaks(-signal, distance=distance)
    plateau_peaks, properties = find_peaks(signal, plateau_size=True, distance=distance)
    #print(plateau_peaks)
    plateau_indices = []
    if 'left_edges' in properties and 'right_edges' in properties:
        for left, right in zip(properties['left_edges'], properties['right_edges']):
            center = (left + right) // 2
            plateau_indices.append(center)
    return {
        'peaks': peaks,
        'troughs': troughs,
        'plateau_indices': np.array(plateau_indices),
        'plateau_info': properties
    }

def reconstruct_signal_from_fft(lip0, top_indices):
    fft_values = np.fft.fft(lip0)
    filtered_fft = np.zeros_like(fft_values, dtype=complex)
    for idx in top_indices:
        filtered_fft[idx] = fft_values[idx]
        if idx != 0:
            filtered_fft[-idx] = fft_values[-idx]
    reconstructed = np.fft.ifft(filtered_fft)
    return reconstructed.real

def lip_fft_representatives(lip0, top_n=3):
    """
    Extrait les représentants significatifs d'un vecteur LIP0 en utilisant la transformée de Fourier.

    Parameters:
        lip0 (ndarray): Vecteur de signatures LIP0.
        top_n (int): Nombre de pics les plus significatifs à extraire.

    Returns:
        list: Indices des représentants dans le vecteur LIP0.
    """
    # Appliquer la transformée de Fourier
    fft_values = np.fft.fft(lip0)
    # Calculer les amplitudes
    amplitudes = np.abs(fft_values)
    
    # Ignorer la composante DC (fréquence 0) en la mettant à zéro
    amplitudes[0] = 0

    # Obtenir les indices des 'top_n' amplitudes les plus élevées
    top_indices = np.argsort(amplitudes)[-top_n:]
    
    # Les indices peuvent être symétriques, on garde la partie positive (fréquences basses)
    top_indices = sorted(top_indices)

    return top_indices
#function to re-order feature:
#lip0
def re_order_feature(lf_m, lf_s, lf_t,
                     ci_m=None, ci_s=None, ci_t=None,
                     om_m=None, om_s=None, om_t=None):
    """
    Assemble les features locaux + optionnels (circularity et orientation merit) 
    en un seul vecteur ordonné.

    Tous les paramètres circulaires et d'orientation sont optionnels.
    S'ils sont absents (None), ils ne sont pas ajoutés au vecteur.
    """
    sizeOfLocalFeature = 3
    feature_ordered = []

    # Concaténation par direction : max/min/median de m, s, t
    assert(len(lf_m) == len(lf_s) == len(lf_t))
    for i in range(0, len(lf_m), sizeOfLocalFeature):
        feature_ordered = np.append(feature_ordered, lf_m[i:i+sizeOfLocalFeature])
        feature_ordered = np.append(feature_ordered, lf_s[i:i+sizeOfLocalFeature])
        feature_ordered = np.append(feature_ordered, lf_t[i:i+sizeOfLocalFeature])

    #--> pour virer lip0 max de chaque image
    #indToDel=[0,3,6]
    #feature_ordered = np.delete(feature_ordered,indToDel)
        
    # Ajout de circularity si fournie
    if ci_m is not None and ci_s is not None and ci_t is not None:
        feature_ordered = np.append(feature_ordered, [ci_m, ci_s, ci_t])

    # Ajout de orientation merit si fourni
    if om_m is not None and om_s is not None and om_t is not None:
        feature_ordered = np.append(feature_ordered, [om_m, om_s, om_t])

    

    return feature_ordered

def circularity(profile):
    img = cv2.imread(profile, 0)

    img = area_closing(img, 128, connectivity=1)
    regions = regionprops(img)
    pe = perimeter_crofton(img, directions=2)
    a = regions[0].area

    ci=(4*math.pi*a)/(pe*pe)
    return ci

def orientation_merit(lip):
    #global FILE STRUCTURE
    sdo=np.max(lip[:, 0])
    #orientation merits
    orientation_merits=1-(math.exp(1-sdo))
    return orientation_merits

def main(argv):
    parser = argparse.ArgumentParser(description="Feature extraction from LIP profile images.")
    parser.add_argument("feature_m", type=str)
    parser.add_argument("feature_s", type=str)
    parser.add_argument("feature_t", type=str)
    parser.add_argument("profile_m", type=str)
    parser.add_argument("profile_s", type=str)
    parser.add_argument("profile_t", type=str)
    parser.add_argument("--use_circularity", action="store_true", help="Include circularity in features")
    parser.add_argument("--use_orientation_merit", action="store_true", help="Include orientation merit in features")
    args = parser.parse_args(argv)
    #path to CSV
    feature_m=argv[0]
    feature_s=argv[1]
    feature_t=argv[2]
    #path to PGM
    profile_m=argv[3]
    profile_s=argv[4]
    profile_t=argv[5]

    #path to CSV
    #feature_m="/volWork/these/DATA/ModelNet/lip10/toilet/test/carac/toilet_0361_m.csv"
    #feature_s="/volWork/these/DATA/ModelNet/lip10/toilet/test/carac/toilet_0361_s.csv"
    #feature_t="/volWork/these/DATA/ModelNet/lip10/toilet/test/carac/toilet_0361_t.csv"
    #path to PGM
    #profile_m="/volWork/these/DATA/ModelNet/lip10/toilet/test/pgm/toilet_0361_m.pgm"
    #profile_s="/volWork/these/DATA/ModelNet/lip10/toilet/test/pgm/toilet_0361_s.pgm"
    #profile_t="/volWork/these/DATA/ModelNet/lip10/toilet/test/pgm/toilet_0361_t.pgm"

    #READ DATA
    df_f_m = pd.read_csv(feature_m,header=None).to_numpy()
    df_f_s = pd.read_csv(feature_s,header=None).to_numpy()
    df_f_t = pd.read_csv(feature_t,header=None).to_numpy()
    #Compute local features
    lf_m=local_features(df_f_m, stats=("max", "min", "median"), mode="default")
    lf_s=local_features(df_f_s, stats=("max", "min", "median"), mode="default")
    lf_t=local_features(df_f_t, stats=("max", "min", "median"), mode="default")

    # Variables optionnelles
    ci_m = ci_s = ci_t = None
    om_m = om_s = om_t = None

    #Circulatiry
    if args.use_circularity:
        ci_m=circularity(profile_m)
        ci_s=circularity(profile_s)
        ci_t=circularity(profile_t)
    #orientation Merits 
    if args.use_orientation_merit:
        om_m=orientation_merit(df_f_m)
        om_s=orientation_merit(df_f_s)
        om_t=orientation_merit(df_f_t)

    #order final feature file
    feature_ordered=re_order_feature(lf_m,lf_s,lf_t,ci_m,ci_s,ci_t,om_m,om_s,om_t)
    #np.set_printoptions(precision=7,suppress=True,linewidth=np.inf)
    feature_ordered = np.round(feature_ordered, decimals=6)
    
    print(*feature_ordered)

if __name__ == "__main__":
    main(sys.argv[1:])
