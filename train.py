# ============================================================================
# WATTIYA - Entraînement du modèle de détection d'anomalies
# ============================================================================
# Cible : type_anomalie (normal / surconsommation / coupure)
# Modèle : RandomForestClassifier
# Sortie : model.joblib + label_encoder.joblib + feature_columns.joblib
# ============================================================================

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)

def feature_engineering(df):
    """
    Crée des features dérivées qui encodent le contexte métier :
    - patterns horaires (cours / soir / nuit)
    - interactions puissance × contexte temporel
    - cyclicité de l'heure
    """
    import numpy as np
    df = df.copy() 

    # 1. Patterns horaires (rythme étudiant)
    df['est_heure_pic_soir']    = df['heure_num'].between(18, 23).astype(int)
    df['est_heure_classe_jour'] = df['heure_num'].between(8, 17).astype(int)
    df['est_nuit_profonde']     = df['heure_num'].between(1, 5).astype(int)

    # 2. Encodage cyclique de l'heure
    #    (le modèle "voit" que 23h et 0h sont proches)
    df['heure_sin'] = np.sin(2 * np.pi * df['heure_num'] / 24)
    df['heure_cos'] = np.cos(2 * np.pi * df['heure_num'] / 24)

    # 3. INTERACTIONS — c'est là que ta connaissance métier passe
    df['puissance_x_nuit']        = df['puissance_w'] * df['est_nuit_profonde']
    df['puissance_x_heure_classe']= df['puissance_w'] * df['est_heure_classe_jour']
    df['puissance_x_weekend']     = df['puissance_w'] * df['est_weekend']

    # 4. Normalisations par contexte
    df['puissance_par_lit']       = df['puissance_w'] / df['nb_lits'].replace(0, 1)

    # 5. Interaction saison × conso
    saison_coef = df['saison'].map({'Hiver': 1.0, 'Automne': 0.7, 'Printemps': 0.6, 'Ete': 0.5})
    df['conso_relative_saison']   = df['puissance_w'] * saison_coef

    return df

# ----------------------------------------------------------------------------
# 0. CHEMINS
# ----------------------------------------------------------------------------
DATA_PATH      = Path('data/dataset_smartenergy.csv')
ARTIFACTS_DIR  = Path('artifacts')
ARTIFACTS_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------------------
# 1. CHARGEMENT
# ----------------------------------------------------------------------------
print(" Chargement du dataset...")
df = pd.read_csv(DATA_PATH, sep=';')
print(f"   {len(df)} lignes, {df.shape[1]} colonnes")

print(" Chargement du dataset...")
df = pd.read_csv(DATA_PATH, sep=';')
print(f"   {len(df)} lignes, {df.shape[1]} colonnes")

# ⬇ NOUVEAU
print("Feature engineering...")
df = feature_engineering(df)
print(f"   Après FE : {df.shape[1]} colonnes")

# ----------------------------------------------------------------------------
# 2. PRÉPROCESSING
# ----------------------------------------------------------------------------
print("\n🧹 Préprocessing...")

# 2a. On supprime timestamp (déjà exploité via heure_num, jour_semaine, saison)
df = df.drop(columns=['timestamp'])

# 2b. On supprime niveau_anomalie (notre cible est type_anomalie)
df = df.drop(columns=['niveau_anomalie'])

# 2c. Encodage one-hot des variables catégorielles
#     RandomForest n'accepte pas les strings → on crée des colonnes 0/1
categorical_cols = ['jour_semaine', 'saison', 'region']
df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

print(f"   Après one-hot encoding : {df.shape[1]} colonnes")

# ----------------------------------------------------------------------------
# 3. SÉPARATION FEATURES / CIBLE
# ----------------------------------------------------------------------------
y = df['type_anomalie']
X = df.drop(columns=['type_anomalie'])

# Encodage de la cible (strings → entiers pour sklearn)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nClasses : {list(label_encoder.classes_)}")
print(f"   Distribution :")
for cls, count in zip(label_encoder.classes_, np.bincount(y_encoded)):
    print(f"     {cls:20s} {count:5d}  ({count/len(y)*100:5.1f}%)")

# Sauvegarde de l'ordre des colonnes (CRUCIAL pour l'inférence)
feature_columns = X.columns.tolist()

# ----------------------------------------------------------------------------
# 4. TRAIN / TEST SPLIT (stratifié)
# ----------------------------------------------------------------------------
print("\nSplit train/test 80/20 stratifié...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded,
    test_size=0.20,
    stratify=y_encoded,    # garde la proportion des classes
    random_state=42
)
print(f"   Train : {len(X_train)} | Test : {len(X_test)}")

# ----------------------------------------------------------------------------
# 5. ENTRAÎNEMENT RANDOM FOREST
# ----------------------------------------------------------------------------
print("\n Entraînement RandomForest...")

model = RandomForestClassifier(
    n_estimators=200,           # 200 arbres
    max_depth=15,               # limite la complexité (évite overfit)
    min_samples_split=5,        # min 5 échantillons pour split
    min_samples_leaf=2,         # min 2 échantillons par feuille
    class_weight='balanced',    #  compense le déséquilibre coupure (6%)
    random_state=42,
    n_jobs=-1                   # utilise tous les cœurs CPU
)

model.fit(X_train, y_train)
print("   ✅ Entraînement terminé")

# ----------------------------------------------------------------------------
# 6. ÉVALUATION
# ----------------------------------------------------------------------------
print("\n📊 Évaluation...")

y_pred = model.predict(X_test)

# Classification report (precision, recall, F1 par classe)
print("\nClassification report :")
print(classification_report(
    y_test, y_pred,
    target_names=label_encoder.classes_,
    digits=3
))

# Matrice de confusion
print("Matrice de confusion :")
cm = confusion_matrix(y_test, y_pred)
print(f"          {'  '.join(f'{c:>14s}' for c in label_encoder.classes_)}")
for i, cls in enumerate(label_encoder.classes_):
    print(f"  {cls:10s}{'  '.join(f'{v:>14d}' for v in cm[i])}")

# Score F1 macro
f1_macro = f1_score(y_test, y_pred, average='macro')
f1_weighted = f1_score(y_test, y_pred, average='weighted')
print(f"\nF1 macro    : {f1_macro:.4f}")
print(f"F1 weighted : {f1_weighted:.4f}")

# Cross-validation (5 folds) pour vérifier la stabilité
print("\n🔁 Cross-validation 5-fold...")
cv_scores = cross_val_score(model, X, y_encoded, cv=5, scoring='f1_macro', n_jobs=-1)
print(f"   F1 macro CV : {cv_scores.mean():.4f} (± {cv_scores.std():.4f})")

# ----------------------------------------------------------------------------
# 7. FEATURE IMPORTANCE
# ----------------------------------------------------------------------------
print("\n🔍 Top 10 features les plus importantes :")
importances = pd.Series(model.feature_importances_, index=feature_columns)
for feat, imp in importances.nlargest(10).items():
    print(f"   {feat:30s} {imp:.4f}")

# ----------------------------------------------------------------------------
# 8. SAUVEGARDE
# ----------------------------------------------------------------------------
print("\n💾 Sauvegarde dans artifacts/ ...")

joblib.dump(model,           ARTIFACTS_DIR / 'model.joblib')
joblib.dump(label_encoder,   ARTIFACTS_DIR / 'label_encoder.joblib')
joblib.dump(feature_columns, ARTIFACTS_DIR / 'feature_columns.joblib')

print("   ✅ model.joblib")
print("   ✅ label_encoder.joblib")
print("   ✅ feature_columns.joblib")
print("\n🎉 Entraînement terminé.")