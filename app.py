# ============================================================================
# WATTIYA ML SERVICE - Microservice Flask de prédiction
# ============================================================================
# Endpoints :
#   GET  /health   → ping
#   POST /predict  → type_anomalie + niveau_anomalie + scores
# ============================================================================

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
from pathlib import Path


# ----------------------------------------------------------------------------
# FEATURE ENGINEERING — DOIT être identique à celui du notebook
# ----------------------------------------------------------------------------
def feature_engineering(df):
    df = df.copy()

    # 1. Patterns horaires (rythme étudiant)
    df['est_heure_pic_soir']    = df['heure_num'].between(18, 23).astype(int)
    df['est_heure_classe_jour'] = df['heure_num'].between(8, 17).astype(int)
    df['est_nuit_profonde']     = df['heure_num'].between(1, 5).astype(int)

    # 2. Encodage cyclique
    df['heure_sin'] = np.sin(2 * np.pi * df['heure_num'] / 24)
    df['heure_cos'] = np.cos(2 * np.pi * df['heure_num'] / 24)

    # 3. Interactions puissance × contexte
    df['puissance_x_nuit']         = df['puissance_w'] * df['est_nuit_profonde']
    df['puissance_x_heure_classe'] = df['puissance_w'] * df['est_heure_classe_jour']
    df['puissance_x_weekend']      = df['puissance_w'] * df['est_weekend']

    # 4. Normalisation par nombre de lits
    df['puissance_par_lit'] = df['puissance_w'] / df['nb_lits'].replace(0, 1)

    # 5. Interaction saison × conso
    saison_coef = df['saison'].map({'Hiver': 1.0, 'Automne': 0.7, 'Printemps': 0.6, 'Ete': 0.5})
    df['conso_relative_saison'] = df['puissance_w'] * saison_coef

    return df


# ----------------------------------------------------------------------------
# CHARGEMENT DES ARTEFACTS AU DÉMARRAGE
# ----------------------------------------------------------------------------
ARTIFACTS_DIR = Path('artifacts')

print("  Chargement du modèle multi-output...")
model            = joblib.load(ARTIFACTS_DIR / 'model.joblib')
le_type          = joblib.load(ARTIFACTS_DIR / 'le_type.joblib')
le_niveau        = joblib.load(ARTIFACTS_DIR / 'le_niveau.joblib')
feature_columns  = joblib.load(ARTIFACTS_DIR / 'feature_columns.joblib')

print(f" Modèle chargé : {len(feature_columns)} features")
print(f"   Classes type   : {list(le_type.classes_)}")
print(f"   Classes niveau : {list(le_niveau.classes_)}")


# ----------------------------------------------------------------------------
# APP FLASK
# ----------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.get('/health')
def health():
    return jsonify({
        'status':            'ok',
        'model':             'RandomForest multi-output',
        'classes_type':      list(le_type.classes_),
        'classes_niveau':    list(le_niveau.classes_),
        'features_expected': len(feature_columns),
    })


@app.post('/predict')
def predict():
    try:
        data = request.get_json(force=True)
        row = pd.DataFrame([data])

        # 1. Feature engineering (identique au training)
        row = feature_engineering(row)

        # 2. One-hot encoding (identique au training)
        row = pd.get_dummies(row, columns=['jour_semaine', 'saison', 'region'], drop_first=False)

        # 3. Réalignement strict sur les colonnes du training
        for col in feature_columns:
            if col not in row.columns:
                row[col] = 0
        row = row[feature_columns]

        # 4. Prédiction MULTI-OUTPUT
        #    model.predict_proba(row) → liste de 2 arrays
        #    [0] = probas pour type_anomalie  (shape: 1, 3)
        #    [1] = probas pour niveau_anomalie (shape: 1, 5)
        probas_list = model.predict_proba(row)
        probas_type   = probas_list[0][0]
        probas_niveau = probas_list[1][0]

        # 5. Décodage des classes prédites
        idx_type   = int(np.argmax(probas_type))
        idx_niveau = int(np.argmax(probas_niveau))

        label_type   = le_type.inverse_transform([idx_type])[0]
        label_niveau = le_niveau.inverse_transform([idx_niveau])[0]

        score_type   = float(probas_type[idx_type])
        score_niveau = float(probas_niveau[idx_niveau])

        # 6. Détail des probabilités par classe
        proba_type_detail = {
            cls: float(probas_type[i])
            for i, cls in enumerate(le_type.classes_)
        }
        proba_niveau_detail = {
            cls: float(probas_niveau[i])
            for i, cls in enumerate(le_niveau.classes_)
        }

        # 7. Top 5 features importantes (global feature importance)
        importances = sorted(
            zip(feature_columns, model.feature_importances_),
            key=lambda x: x[1],
            reverse=True
        )[:5]

        # 8. Réponse normalisée pour le backend Node
        return jsonify({
            'type_anomalie':    label_type,
            'niveau_priorite':  label_niveau,
            'score_type':       score_type,
            'score_niveau':     score_niveau,
            'score_ia':         (score_type + score_niveau) / 2,   # score global
            'probabilities': {
                'type':   proba_type_detail,
                'niveau': proba_niveau_detail
            },
            'top_features': [
                {'feature': name, 'importance': float(imp)} 
                for name, imp in importances
            ],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)