
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os
import sys

def lip_fft_representatives(lip0, top_n=5):
    fft_values = np.fft.fft(lip0)
    amplitudes = np.abs(fft_values)
    amplitudes[0] = 0
    top_indices = np.argsort(amplitudes)[-top_n:]
    return sorted(top_indices), amplitudes

def reconstruct_signal_from_fft(lip0, top_indices):
    fft_values = np.fft.fft(lip0)
    filtered_fft = np.zeros_like(fft_values, dtype=complex)
    for idx in top_indices:
        filtered_fft[idx] = fft_values[idx]
        if idx != 0:
            filtered_fft[-idx] = fft_values[-idx]
    reconstructed = np.fft.ifft(filtered_fft)
    return reconstructed.real

def detect_signal_features(signal, distance=10):
    peaks, _ = find_peaks(signal, distance=distance)
    troughs, _ = find_peaks(-signal, distance=distance)
    plateau_peaks, properties = find_peaks(signal, plateau_size=True, distance=distance)
    print(plateau_peaks)
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

def select_lip_representative_indices(lip0, top_n_freq=5, n_representatives=10):
    """
    Sélectionne un nombre fixe de représentants à partir de la reconstruction FFT de lip0,
    en répartissant de façon équilibrée les pics, creux et plateaux.
    Complète si nécessaire avec les points les plus extrêmes.
    """
    top_indices, _ = lip_fft_representatives(lip0, top_n=top_n_freq)
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


def visualize_reconstructed_with_annotated_points(reconstructed, representative_indices, features, save_prefix):
    peaks = features['peaks']
    troughs = features['troughs']
    plateau_indices = features['plateau_indices']

    categories = {'peak': [], 'trough': [], 'plateau': [], 'other': []}
    for idx in representative_indices:
        if idx in peaks:
            categories['peak'].append(idx)
        elif idx in troughs:
            categories['trough'].append(idx)
        elif idx in plateau_indices:
            categories['plateau'].append(idx)
        else:
            categories['other'].append(idx)

    plt.figure(figsize=(10, 4))
    plt.plot(reconstructed, label='Signal reconstruit (FFT)', color='black', linewidth=1.5)

    if categories['peak']:
        plt.scatter(categories['peak'], reconstructed[categories['peak']], marker='^', color='red', label='Pics')
    if categories['trough']:
        plt.scatter(categories['trough'], reconstructed[categories['trough']], marker='v', color='blue', label='Creux')
    if categories['plateau']:
        plt.scatter(categories['plateau'], reconstructed[categories['plateau']], marker='s', color='orange', label='Plateaux')
    if categories['other']:
        plt.scatter(categories['other'], reconstructed[categories['other']],
                    facecolors='none', edgecolors='green', marker='D', s=60, label='Compléments')

    plt.title("Signal reconstruit avec représentants annotés")
    plt.xlabel("Index")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_prefix}_reconstructed_annotated.png")
    plt.close()
    print(f"[INFO] Figure annotée sauvegardée : {save_prefix}_reconstructed_annotated.png")

def extract_features_by_indices(lip, indices):
    features = []
    for col in range(lip.shape[1]):
        features.extend(lip[indices, col])
    return np.array(features)

def main(csv_path):
    base_name = os.path.splitext(os.path.basename(csv_path))[0]
    save_prefix = base_name

    lip = pd.read_csv(csv_path, header=None).to_numpy()
    lip0 = lip[:, 0]

    representative_indices, top_indices, reconstructed, features = select_lip_representative_indices(
        lip0, top_n_freq=10, n_representatives=10
    )

    visualize_reconstructed_with_annotated_points(
        reconstructed, representative_indices, features, save_prefix
    )

    #features_vector = extract_features_by_indices(lip, representative_indices)
    #features_output_path = f"{save_prefix}_features.csv"
    #np.savetxt(features_output_path, features_vector[np.newaxis], delimiter=",", fmt="%.6f")
    #print(f"[INFO] Features sauvegardées dans : {features_output_path}")
    #print(f"[INFO] Indices sélectionnés : {representative_indices}")

    #return features_vector, representative_indices

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage : python fft_lip_feature_selector.py chemin/vers/fichier_lip.csv")
        sys.exit(1)

    csv_path = sys.argv[1]
    main(csv_path)