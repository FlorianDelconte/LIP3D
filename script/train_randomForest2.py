import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import json
import sys
import os
import argparse

def save_hyperparameters(params, file_path):
    """Sauvegarde des hyperparamètres dans un fichier JSON."""
    try:
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Hyperparamètres sauvegardés dans : {file_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des hyperparamètres : {e}")

def load_hyperparameters(file_path):
    """Charge les hyperparamètres depuis un fichier JSON."""
    try:
        with open(file_path, 'r') as f:
            params = json.load(f)
        print(f"Hyperparamètres chargés depuis : {file_path}")
        return params
    except Exception as e:
        print(f"Erreur lors du chargement des hyperparamètres : {e}")
        return None

def perform_grid_search(X_train, y_train):
    """Effectue un GridSearch pour trouver les meilleurs hyperparamètres."""
    print("Lancement du GridSearch...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt']
    }
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print("Meilleurs paramètres trouvés :", grid_search.best_params_)
    return grid_search.best_params_

def ablation_study(X_train, y_train, X_test, y_test, params, class_names, data_path):
    """Effectue une étude d'ablation."""
    print("\n=== Début de l'Ablation Study ===")
    taille_orientation = 30
    taille_lip = 5

    feature_groups = {
        'Orientation_principale': slice(0, taille_orientation),
        'Orientation_secondaire': slice(taille_orientation, 2 * taille_orientation),
        'Orientation_tertiaire': slice(2 * taille_orientation, 3 * taille_orientation),
    }

    for i in range(6):
        indices = np.r_[i*taille_lip:(i+1)*taille_lip,
                        taille_orientation + i*taille_lip:taille_orientation + (i+1)*taille_lip,
                        2*taille_orientation + i*taille_lip:2*taille_orientation + (i+1)*taille_lip]
        feature_groups[f'LIP{i}_all_orientations'] = indices

    results_ablation = {}
    for name, indices in feature_groups.items():
        print(f"\nTesting: {name}")
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train.iloc[:, indices], y_train)
        y_pred_subset = model.predict(X_test.iloc[:, indices])
        report_subset = classification_report(y_test, y_pred_subset, output_dict=True, zero_division=0.0, target_names=class_names)
        results_ablation[name] = {cls: report_subset[cls]['f1-score'] for cls in class_names}

    results_df = pd.DataFrame(results_ablation).T
    print("\nRésultats détaillés de l'Ablation Study (F1-score par classe) :")
    print(results_df)

    plt.figure(figsize=(12, 8))
    sns.heatmap(results_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
    plt.title('Ablation Study - F1-score par classe')
    plt.xlabel('Classe')
    plt.ylabel('Combinaison de caractéristiques')
    plt.tight_layout()
    plt.savefig(data_path + 'ablation_study_results.png')

# Parsing des arguments
parser = argparse.ArgumentParser(description="Entraînement et ablation avec Random Forest.")
parser.add_argument('-g', '--gridsearch', action='store_true', help="Effectuer le GridSearch pour les hyperparamètres.")
parser.add_argument('-a', '--ablation', action='store_true', help="Effectuer l'Ablation Study.")
parser.add_argument('-p', '--params', type=str, help="Fichier JSON contenant les hyperparamètres.")
args = parser.parse_args()

# Chemins vers les fichiers
data_path = "/volWork/these/DATA/ModelNet/lipCustom10/EXP/3lipOm/"
train_features = pd.read_csv(data_path + 'train_caracs.txt', header=None, delimiter=" ")
train_labels = pd.read_csv(data_path + 'train_labels.txt', header=None, names=['labels'])
test_features = pd.read_csv(data_path + 'test_caracs.txt', header=None, delimiter=" ")
test_labels = pd.read_csv(data_path + 'test_labels.txt', header=None, names=['labels'])
#class name pour lipCustom10
class_names = ['airplane', 'bed', 'car', 'cone', 'door', 'glass_box', 'guitar', 'monitor', 'table', 'toilet']
#class name pour lipC40
#class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
#               'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
#               'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
#               'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
# Chargement ou calcul des hyperparamètres
params = {}
if args.params:
    params = load_hyperparameters(args.params)
if not params and args.gridsearch:
    params = perform_grid_search(train_features, train_labels.values.ravel())

if not params:
    print("Aucun hyperparamètre valide, utilisez l'option -g pour lancer GridSearch.")
    sys.exit(1)

# Entraînement du modèle
#params.pop('random_state')
model = RandomForestClassifier(**params, random_state=42)
model.fit(train_features, train_labels.values.ravel())
joblib.dump(model, data_path + 'best_model_rf.joblib')

# Étude d'ablation si demandée
if args.ablation:
    ablation_study(train_features, train_labels.values.ravel(), test_features, test_labels.values.ravel(), params, class_names, data_path)
