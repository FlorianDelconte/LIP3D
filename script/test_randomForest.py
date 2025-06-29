import joblib
import sys
import os
import json
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

def load_and_display_model(model_path):
    """
    Charge un modèle Random Forest et affiche ses hyperparamètres.

    Parameters:
        model_path (str): Chemin vers le modèle sauvegardé (fichier .joblib).
    """
    # Vérification de l'existence du fichier
    if not os.path.isfile(model_path):
        print(f"Erreur : Le fichier '{model_path}' n'existe pas.")
        sys.exit(1)

    try:
        # Chargement du modèle
        model = joblib.load(model_path)
        print(f"Modèle chargé depuis : {model_path}\n")
        
        # Vérification si le modèle est bien un RandomForestClassifier
        if not hasattr(model, 'get_params'):
            print("Erreur : Le modèle chargé n'est pas un Random Forest.")
            sys.exit(1)

        # Affichage des hyperparamètres
        print("Hyperparamètres du modèle Random Forest :")
        for param, value in model.get_params().items():
            print(f"  {param}: {value}")
        
        return model

    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {str(e)}")
        sys.exit(1)

def test_model(model, test_features_path, test_labels_path, class_names, save_path=None):
    import seaborn as sns  # Assure qu'on a seaborn

    try:
        X_test = pd.read_csv(test_features_path, header=None, delimiter=" ", dtype=np.float32)
        y_test = pd.read_csv(test_labels_path, header=None, delimiter=" ", names=['labels'])
    except Exception as e:
        print(f"Erreur lors du chargement des données de test : {e}")
        sys.exit(1)

    print("Données de test chargées avec succès.")

    y_pred = model.predict(X_test)

    print("\n=== Rapport de classification ===")
    report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    print(report)

    conf_matrix = confusion_matrix(y_test, y_pred)
    conf_matrix_norm = conf_matrix.astype('float') / conf_matrix.sum(axis=1, keepdims=True)

    # Tracer avec seaborn (couleur = proportion, texte = valeur brute)
    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(conf_matrix_norm, annot=conf_matrix, fmt='d',
                     xticklabels=class_names, yticklabels=class_names,
                     cmap='Blues', cbar_kws={'label': 'Proportion (normalisée par ligne)'})

    plt.xlabel("Classe prédite")
    plt.ylabel("Classe réelle")
    plt.title("Matrice de confusion\n(couleur = proportion, valeur = comptage)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"[INFO] Matrice de confusion sauvegardée dans : {save_path}")
    else:
        plt.show()



if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Charger et tester un modèle Random Forest")
    parser.add_argument("model_path", type=str, help="Chemin vers le modèle (.joblib)")
    parser.add_argument("-t", "--test_data", nargs=2, metavar=("features", "labels"), help="Chemins vers les fichiers de test")
    parser.add_argument("-c", "--classes", type=str, nargs="+", help="Liste des noms de classes")
    args = parser.parse_args()

    # Chargement du modèle
    model = load_and_display_model(args.model_path)

    # Si des données de test sont fournies, effectuer le test
    if args.test_data:
        if not args.classes:
            print("Erreur : La liste des noms de classes est requise pour tester le modèle.")
            sys.exit(1)
        
        test_features_path, test_labels_path = args.test_data
        class_names = args.classes

        # Test du modèle
        test_model(model, test_features_path, test_labels_path, class_names, "/volWork/these/DATA/ModelNet/lipCustom10/EXP/2lipOm/confusion_matrix.png")

    else:
        print("Aucun test effectué. Pour tester, utilisez l'option -t avec les chemins des données.")
