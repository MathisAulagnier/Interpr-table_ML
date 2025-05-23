import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def _get_pseudo_labels_hard(X: pd.DataFrame, blackbox_model) -> np.ndarray:
    """
    Génère des pseudo-labels durs à partir du modèle black-box.
    """
    if hasattr(blackbox_model, 'predict_proba'):
        proba = blackbox_model.predict_proba(X)
        return np.argmax(proba, axis=1)
    elif hasattr(blackbox_model, 'predict'):
        return blackbox_model.predict(X)
    else:
        raise ValueError("Le modèle doit avoir 'predict' ou 'predict_proba'.")

def distill_with_logistic_regression(X: pd.DataFrame, blackbox_model) -> LogisticRegression:
    """
    Distillation par Régression Logistique.
    """
    y_distilled = _get_pseudo_labels_hard(X, blackbox_model)
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X, y_distilled)
    return model

def distill_with_decision_tree(X: pd.DataFrame, blackbox_model, max_depth: int = 3) -> DecisionTreeClassifier:
    """
    Distillation par Arbre de Décision peu profond.
    """
    y_distilled = _get_pseudo_labels_hard(X, blackbox_model)
    model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    model.fit(X, y_distilled)
    return model