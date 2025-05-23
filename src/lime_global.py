import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from lime.lime_tabular import LimeTabularExplainer
except ImportError:
    LimeTabularExplainer = None

def global_lime_weights(
    X: pd.DataFrame,
    blackbox_model,
    num_samples: int = 100
) -> pd.Series:
    """
    Calcule l'importance globale des features en agrégeant des explications LIME locales.
    """
    if LimeTabularExplainer is None or not hasattr(blackbox_model, 'predict_proba'):
        return pd.Series(dtype=float)

    X_np = X.values
    names = X.columns.tolist()
    explainer = LimeTabularExplainer(
        training_data=X_np,
        feature_names=names,
        class_names=[str(c) for c in getattr(blackbox_model, 'classes_', [])],
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )

    weights_sum = np.zeros(len(names))
    # échantillonnage
    idx = np.random.choice(X.shape[0], min(num_samples, X.shape[0]), replace=False)
    for i in idx:
        exp = explainer.explain_instance(
            X_np[i],
            blackbox_model.predict_proba,
            num_features=len(names)
        )
        label = exp.top_labels[0] if exp.top_labels else 0
        local = exp.local_exp.get(label, [])
        for feat_idx, w in local:
            weights_sum[feat_idx] += abs(w)

    avg = weights_sum / max(1, len(idx))
    return pd.Series({n: avg[i] for i, n in enumerate(names)}).sort_values(ascending=False)

def plot_lime_weights(global_weights: pd.Series) -> plt.Figure:
    """
    Barplot des poids LIME moyens.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(global_weights) * 0.3)))
    sns.barplot(x=global_weights.values, y=global_weights.index, ax=ax)
    ax.set_xlabel("Poids LIME moyens")
    ax.set_ylabel("Feature")
    ax.set_title("Importance globale (LIME aggregé)")
    fig.tight_layout()
    return fig