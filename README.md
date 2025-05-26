# Application d'Interprétabilité de Modèles par Distillation

## 🎯 Objectif

Cette application web, développée avec Streamlit, a pour but de rendre les modèles de Machine Learning complexes (souvent appelés "boîtes noires") plus compréhensibles. Pour ce faire, elle utilise une technique d'**interprétabilité appelée distillation de connaissances**. L'idée est d'entraîner un modèle plus simple et intrinsèquement interprétable (le "modèle étudiant") pour imiter le comportement d'un modèle "boîte noire" pré-entraîné (le "modèle professeur").

Actuellement, l'application se concentre sur :
* L'utilisation d'un modèle classifieur scikit-learn compatible en tant que modèle professeur.
* L'entraînement d'un **Arbre de Décision (DecisionTreeClassifier)** comme modèle étudiant.

## ✨ Fonctionnalités Principales

* **Téléversement Facile** :
    * Chargez votre propre modèle professeur scikit-learn au format `.joblib`.
    * Chargez votre jeu de données (features uniquement) au format `.csv` sur lequel le modèle professeur a été entraîné ou peut faire des prédictions.
* **Processus de Distillation** :
    * Le modèle professeur génère des prédictions (étiquettes) sur les données fournies.
    * Un modèle Arbre de Décision étudiant est ensuite entraîné en utilisant ces prédictions comme cibles.
* **Évaluation de la Fidélité** :
    * Mesurez à quel point le modèle étudiant parvient à imiter les prédictions du modèle professeur grâce à un score d'accuracy (fidélité).
* **Visualisation Interprétable** :
    * Visualisez l'Arbre de Décision étudiant résultant, vous permettant de comprendre les règles de décision qu'il a apprises.
* **Interface Utilisateur Intuitive** :
    * Une interface simple et conviviale construite avec Streamlit, permettant d'ajuster certains paramètres comme la profondeur maximale de l'arbre étudiant.
* **Téléchargement des Modèles** :
    * Téléchargez le modèle étudiant entraîné et les prédictions du modèle professeur pour une utilisation ultérieure.

## 🤔 Pourquoi la Distillation ?

Les modèles modernes de Machine Learning (par exemple, les forêts aléatoires, les réseaux de neurones profonds) peuvent atteindre des performances très élevées, mais leur complexité interne les rend difficiles à interpréter. Comprendre *pourquoi* un modèle prend une décision particulière est crucial dans de nombreux domaines (santé, finance, justice).

La distillation de connaissances permet de :
1.  Créer une **approximation simplifiée** du modèle complexe.
2.  Obtenir un **modèle interprétable** qui, s'il est suffisamment fidèle, peut donner des indications sur le fonctionnement du modèle original.
3.  Déployer potentiellement un modèle plus léger (l'étudiant) si sa performance est acceptable.

## 🛠️ Technologies Utilisées

* **Python** : Langage de programmation principal.
* **Streamlit** : Framework pour la création rapide d'applications web pour la data science.
* **Scikit-learn** : Bibliothèque pour le Machine Learning (modèles, métriques).
* **Pandas** : Bibliothèque pour la manipulation et l'analyse de données.
* **Joblib** : Pour la sérialisation/désérialisation des modèles scikit-learn.
* **Matplotlib** : Pour la création de visualisations (notamment l'arbre de décision).

## 🚀 Comment Lancer l'Application

1.  **Prérequis** :
    * Assurez-vous d'avoir Python (3.7+ recommandé) et pip installés.


    * Placez le fichier `app_distillation.py` (et potentiellement les fichiers d'exemple `teacher_model.joblib` et `sample_features.csv`) dans un répertoire de projet.
    *  **Installer les Dépendances** :

    ```bash
    pip install streamlit pandas scikit-learn joblib matplotlib
    ```

4.  **Préparer vos Fichiers (Optionnel - pour tester avec les exemples)** :
    * Un **modèle professeur** au format `.joblib` (ex: `teacher_model.joblib`). Ce modèle doit être un classifieur scikit-learn entraîné.
    * Un **fichier CSV de features** (ex: `sample_features.csv`). Ce fichier doit contenir les colonnes de features que votre modèle professeur attend, sans la colonne cible originale.

    *Vous pouvez générer des fichiers d'exemple en utilisant le script `create_teacher_model.py`.*

5.  **Exécuter l'Application Streamlit** :
    Toujours dans le terminal, à la racine de votre projet, lancez :
    ```bash
    streamlit run app_distillation.py
    ```
    L'application devrait s'ouvrir automatiquement dans votre navigateur web par défaut.