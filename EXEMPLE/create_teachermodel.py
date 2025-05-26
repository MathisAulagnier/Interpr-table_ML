# Script pour créer un modèle exemple (create_teacher_model.py)
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
import pandas as pd
import joblib

# Générer des données synthétiques
X_data, y_data = make_classification(n_samples=200, n_features=5, n_informative=3, n_redundant=0, random_state=42, n_classes=2)
feature_names = [f'feature_{i}' for i in range(X_data.shape[1])]
X_df = pd.DataFrame(X_data, columns=feature_names)

# Entraîner un RandomForestClassifier
teacher = RandomForestClassifier(n_estimators=100, random_state=42)
teacher.fit(X_df, y_data)

# Sauvegarder le modèle
model_filename = 'teacher_model.joblib'
joblib.dump(teacher, model_filename)
print(f"Modèle professeur sauvegardé sous : {model_filename}")

# Sauvegarder les features (sans les étiquettes y_data) pour le test de l'application
# L'application s'attend à un CSV avec uniquement les features
features_filename = 'sample_features.csv'
X_df.to_csv(features_filename, index=False)
print(f"Fichier de features CSV sauvegardé sous : {features_filename}")

# (Optionnel) Sauvegarder les données complètes si vous voulez vérifier les prédictions
# full_data_df = X_df.copy()
# full_data_df['target'] = y_data
# full_data_df.to_csv('sample_full_data.csv', index=False)