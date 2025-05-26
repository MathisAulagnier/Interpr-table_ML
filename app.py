import streamlit as st
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier 
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import io 

# --- Fonctions Utilitaires ---
def load_model_from_upload(uploaded_file):
    """Charge un modèle scikit-learn depuis un fichier uploadé (.joblib)."""
    if uploaded_file is not None:
        try:
            file_content = uploaded_file.getvalue()
            # Charger le modèle depuis le contenu en mémoire
            model = joblib.load(io.BytesIO(file_content))
            return model
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle : {e}")
            return None
    return None

def load_data_from_upload(uploaded_file):
    """Charge des données depuis un fichier CSV uploadé."""
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Erreur lors du chargement du fichier CSV : {e}")
            return None
    return None

# --- Configuration de la Page Streamlit ---
st.set_page_config(layout="wide", page_title="Distillation de Modèle Simple")

# --- Titre de l'Application ---
st.title("GlassBoxAI : la distillation d'un modèle 'Boîte Noire'")
st.markdown("""
Cette application permet de distiller les connaissances d'un modèle complexe (le "professeur")
vers un modèle plus simple et interprétable (l'"étudiant", ici un Arbre de Décision).

**Instructions :**
1.  **Téléversez votre modèle professeur** (`.joblib`): Il doit être un modèle scikit-learn entraîné.
2.  **Téléversez vos données** (`.csv`): Ce fichier CSV doit contenir **uniquement les features** prétraitées, telles qu'attendues par votre modèle professeur. Les noms des colonnes seront utilisés comme noms de features.
""")

# --- Interface Utilisateur (Sidebar pour les uploads) ---
st.sidebar.header("Configuration")
uploaded_teacher_model_file = st.sidebar.file_uploader(
    "1. Téléversez le modèle professeur (.joblib)",
    type=['joblib']
)
uploaded_data_file = st.sidebar.file_uploader(
    "2. Téléversez les données CSV (features uniquement)",
    type=['csv']
)

# Paramètre pour le modèle étudiant
st.sidebar.subheader("Paramètres du modèle étudiant")
student_max_depth = st.sidebar.slider(
    "Profondeur maximale de l'arbre étudiant :",
    min_value=1,
    max_value=15,
    value=3,
    step=1
)

# --- Logique Principale de l'Application ---
if uploaded_teacher_model_file and uploaded_data_file:
    # Charger le modèle et les données
    teacher_model = load_model_from_upload(uploaded_teacher_model_file)
    df_features = load_data_from_upload(uploaded_data_file)

    if teacher_model and df_features is not None:
        st.header("1. Données et modèle 'Professeur'")
        st.subheader("Aperçu des Données (Features) Chargées :")
        st.dataframe(df_features.head())
        st.write(f"Nombre d'échantillons : {df_features.shape[0]}, Nombre de features : {df_features.shape[1]}")

        try:
            # Obtenir les prédictions du modèle professeur (ce seront les "labels" pour l'étudiant)
            st.subheader("Prédictions du Modèle Professeur :")
            teacher_predictions = teacher_model.predict(df_features)
            # Si votre modèle professeur est un classifieur et que vous voulez utiliser les probabilités pour la distillation :
            # teacher_proba_predictions = teacher_model.predict_proba(df_features)

            df_teacher_preds = pd.DataFrame({'Prédictions_Professeur': teacher_predictions})
            st.dataframe(df_teacher_preds.head())
            st.write(f"Nombre de prédictions générées : {len(teacher_predictions)}")

            # Afficher les classes uniques prédites par le professeur
            unique_teacher_preds = pd.Series(teacher_predictions).unique()
            st.write(f"Classes prédites par le professeur : {unique_teacher_preds}")


            st.header("2. Entraînement et Évaluation du Modèle Étudiant")

            # Définir le modèle étudiant (Arbre de Décision)
            student_model = DecisionTreeClassifier(
                max_depth=student_max_depth,
                random_state=42 # Important pour la reproductibilité
            )

            student_model.fit(df_features, teacher_predictions)
            st.success(f"Modèle étudiant (Arbre de Décision avec max_depth={student_max_depth}) entraîné avec succès !")

            # Prédictions du modèle étudiant sur les mêmes données
            student_predictions = student_model.predict(df_features)

            # Évaluer la "fidélité" : à quel point l'étudiant imite bien le professeur
            fidelity = accuracy_score(teacher_predictions, student_predictions)
            st.metric(label="Fidélité (Accuracy Étudiant vs. Professeur)", value=f"{fidelity:.4f}")
            if fidelity < 0.8:
                st.warning("La fidélité est relativement basse. Le modèle étudiant pourrait ne pas bien imiter le professeur. Essayez d'augmenter la profondeur de l'arbre ou d'explorer d'autres types de modèles étudiants.")
            elif fidelity < 0.95:
                st.info("La fidélité est bonne ! Le modèle étudiant semble bien approximer le professeur.")
            else:
                st.success("Excellente fidélité ! Le modèle étudiant imite très bien le professeur.")

            # Bouton pour telecharger le modèle étudiant
            st.write("Le modèle étudiant a été sauvegardé et peut être téléchargé ci-dessous.")
            student_model_filename = "student_model.joblib"
            # joblib.dump(student_model, student_model_filename)
            st.download_button(
                label="Télécharger le modèle étudiant (.joblib)",
                data=io.BytesIO(joblib.dump(student_model, io.BytesIO())),
                file_name=student_model_filename,
                mime="application/octet-stream"
            )

            st.header("3. Interprétation du Modèle Étudiant")
            st.subheader("Visualisation de l'Arbre de Décision Étudiant :")

            # Toggles pour les options de visualisation
            impurity_bool = st.sidebar.toggle("Afficher l'impureté (Gini)", value=False)
            proportion_bool = st.sidebar.toggle("Afficher les proportions dans les feuilles", value=False)
            
            # Visualiser l'arbre de décision étudiant
            if df_features.shape[1] > 0: # S'il y a des features
                feature_names = df_features.columns.tolist()
                class_names_student = [str(c) for c in student_model.classes_]

                try:
                    fig, ax = plt.subplots(figsize=(min(25, 5 + student_max_depth * 3), min(15, 3 + student_max_depth * 2)))
                    plot_tree(student_model,
                              feature_names=feature_names,
                              class_names=class_names_student,
                              filled=True,
                              rounded=True,
                              fontsize=max(6, 12 - student_max_depth), # Ajuster la taille de la police
                              impurity= impurity_bool, # Afficher l'impureté (ex: Gini)
                              proportion= proportion_bool) # Afficher les proportions dans les feuilles
                    st.pyplot(fig)
                    st.caption("Chaque nœud montre : la condition de séparation, l'impureté (gini), le nombre d'échantillons (samples), la distribution des échantillons par classe (value), et la classe majoritaire (class). Les proportions sont affichées si `proportion=True`.")
                except Exception as e:
                    st.error(f"Erreur lors de la création de la visualisation de l'arbre : {e}")
                    st.info("Cela peut arriver si l'arbre est très simple (ex: une seule feuille) ou si les noms de features/classes posent problème.")
            else:
                st.warning("Aucune feature trouvée dans les données pour visualiser l'arbre.")

        except AttributeError as e:
            st.error(f"Erreur avec le modèle professeur : {e}. Assurez-vous qu'il a une méthode 'predict' et qu'il est compatible avec les données fournies.")
            st.error("Vérifiez que le modèle `.joblib` est bien un modèle scikit-learn entraîné et que les features du CSV correspondent.")
        except Exception as e:
            st.error(f"Une erreur générale est survenue lors du traitement : {e}")
            st.error("Veuillez vérifier vos fichiers et les formats attendus.")

elif uploaded_teacher_model_file or uploaded_data_file:
    st.info("Veuillez téléverser à la fois le modèle professeur et le fichier de données CSV pour continuer.")
else:
    st.info("Bienvenue ! Commencez par téléverser votre modèle et vos données via le panneau latéral.")

st.markdown("---")

