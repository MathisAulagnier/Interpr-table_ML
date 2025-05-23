import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

try:
    from imodels import SLIMClassifier
except ImportError:
    SLIMClassifier = None

def train_riskslim_surrogate(X: pd.DataFrame, blackbox_model):
    """
    Entraîne un SLIMClassifier pour approximer RiskSLIM.
    """
    if SLIMClassifier is None:
        return None

    # pseudo-labels
    y = (blackbox_model.predict_proba(X) if hasattr(blackbox_model, 'predict_proba')
         else blackbox_model.predict(X))
    y = np.argmax(y, axis=1) if y.ndim > 1 else y

    model = SLIMClassifier(random_state=42)
    model.fit(X, y)
    return model

def evaluate_accuracy(model, X: pd.DataFrame, y_true) -> float:
    """
    Précision vs véritables labels.
    """
    if model is None: return 0.0
    return accuracy_score(y_true, model.predict(X))

def evaluate_fidelity(model, blackbox_model, X: pd.DataFrame) -> float:
    """
    Fidelity = correspondance aux prédictions du black-box.
    """
    if model is None or blackbox_model is None: return 0.0
    y_bb = (blackbox_model.predict_proba(X) if hasattr(blackbox_model, 'predict_proba')
            else blackbox_model.predict(X))
    y_bb = np.argmax(y_bb, axis=1) if y_bb.ndim > 1 else y_bb
    return accuracy_score(y_bb, model.predict(X))