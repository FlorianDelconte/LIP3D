import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from IPython import display
import matplotlib.pyplot as plt
import joblib
import seaborn as sns
import json

# Sauvegarde des hyperparamètres
def save_hyperparameters(params, file_path):
    try:
        with open(file_path, 'w') as f:
            json.dump(params, f, indent=4)
        print(f"Hyperparamètres sauvegardés dans : {file_path}")
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des hyperparamètres : {e}")

def visu_classif_report(report, class_names, name):
    display.display(pd.DataFrame(report))
    df = pd.DataFrame(report)
    df.iloc[:3, :len(class_names)].T.plot(kind='bar', ylim=(0, 1))
    plt.savefig(name)

# Chemins vers les fichiers
data_path = "/volWork/these/DATA/ModelNet/lip10/EXP/defaut/"

train_featuresPathFileLIP = data_path + 'train_caracs.txt'
train_labelsPathFile = data_path + 'train_labels.txt'

test_featuresPathFileLIP = data_path + 'test_caracs.txt'
test_labelsPathFile = data_path + 'test_labels.txt'
# Sauvegarder les meilleurs paramètres trouvés
param_file = data_path + 'best_rf_params.json'

# Chargement des données d'entraînement
X_train_LIP = pd.read_csv(train_featuresPathFileLIP, header=None, delimiter=" ", dtype=np.float32)
y_train = pd.read_csv(train_labelsPathFile, header=None, delimiter=" ", names=['labels'])

# Chargement des données de test
X_test_LIP = pd.read_csv(test_featuresPathFileLIP, header=None, delimiter=" ", dtype=np.float32)
y_test = pd.read_csv(test_labelsPathFile, header=None, delimiter=" ", names=['labels'])

# Définition unique des noms des classes
class_names = ['bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet']
'''class_names = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone',
               'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp',
               'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink',
               'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox']
'''
# Hyperparamètres
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': [None, 'sqrt','log2']
}

# Initialisation et optimisation du modèle
rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
grid_search.fit(X_train_LIP, y_train.values.ravel())

best_params = grid_search.best_params_
print("Meilleurs paramètres trouvés :", best_params)

best_rf = grid_search.best_estimator_
y_pred = best_rf.predict(X_test_LIP)

save_hyperparameters(best_params, param_file)

# Rapport de classification baseline
report = classification_report(y_test, y_pred, output_dict=True, zero_division=0.0, target_names=class_names)
visu_classif_report(report, class_names, data_path+'classification_report.png')

# Matrice de confusion baseline
conf_matrix = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
disp.plot(cmap='Blues', ax=ax)
plt.title('Matrice de confusion (baseline)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(data_path+'confusion_matrix_baseline.png')

# Ablation study (avec hyperparamètres optimisés)
# Paramètres modifiables
taille_orientation = 30
taille_lip = 5

feature_groups = {
    'Orientation_principale': slice(0, taille_orientation),
    'Orientation_secondaire': slice(taille_orientation, 2 * taille_orientation),
    'Orientation_tertiaire': slice(2 * taille_orientation, 3 * taille_orientation),
}

# Ajout dynamique des LIP pour toutes les orientations
for i in range(6):
    indices = np.r_[i*taille_lip:(i+1)*taille_lip,
                    taille_orientation + i*taille_lip:taille_orientation + (i+1)*taille_lip,
                    2*taille_orientation + i*taille_lip:2*taille_orientation + (i+1)*taille_lip]
    feature_groups[f'LIP{i}_all_orientations'] = indices

results_ablation = {}
for name, indices in feature_groups.items():
    print(f"\nTesting: {name}")
    model = RandomForestClassifier(**best_params, random_state=42)
    model.fit(X_train_LIP.iloc[:, indices], y_train.values.ravel())
    y_pred_subset = model.predict(X_test_LIP.iloc[:, indices])
    report_subset = classification_report(y_test, y_pred_subset, output_dict=True, zero_division=0.0,
                                          target_names=class_names)
    results_ablation[name] = {classe: report_subset[classe]['f1-score'] for classe in class_names}

# Affichage des résultats détaillés
results_df = pd.DataFrame(results_ablation).T
print("\nRésultats détaillés de l'ablation study (F1-score par classe) :")
print(results_df)

# Heatmap des résultats
plt.figure(figsize=(12, 8))
sns.heatmap(results_df, annot=True, cmap='YlGnBu', vmin=0, vmax=1)
plt.title('Ablation Study - F1-score par classe')
plt.xlabel('Classe')
plt.ylabel('Combinaison de caractéristiques')
plt.tight_layout()
plt.savefig(data_path+'ablation_study_results_per_class.png')

# Sauvegarde du meilleur modèle
joblib.dump(best_rf, data_path+'best_model_rf.joblib')
print("Meilleur modèle sauvegardé sous :", data_path+'best_model_rf.joblib')
