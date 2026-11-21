"""The model ladder: from tiny to heavy, the same 10 rungs for every scenario.

Rungs are ordered by expected compute so that the carbon axis is spanned
from ~1e-7 kg (a linear model) to the hyper-parameter search (the most
compute-hungry practice in day-to-day industrial ML).
"""
from __future__ import annotations

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import (HistGradientBoostingClassifier, HistGradientBoostingRegressor,
                              RandomForestClassifier, RandomForestRegressor,
                              StackingClassifier, StackingRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from config import N_JOBS

# (name, family) in ladder order; family is used for colours in the plots
LADDER = [
    ("Linear", "linear"), ("Tree-d6", "tree"), ("RF-100", "bagging"), ("RF-500", "bagging"),
    ("HGB-100", "boosting"), ("HGB-500", "boosting"), ("MLP-S", "neural"), ("MLP-L", "neural"),
    ("Stack", "ensemble"), ("HGB-HPO", "search"),
]


def build(name: str, task: str, seed: int):
    clf = task == "clf"
    if name == "Linear":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)) if clf \
            else make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    if name == "Tree-d6":
        return (DecisionTreeClassifier if clf else DecisionTreeRegressor)(max_depth=6, random_state=seed)
    if name in ("RF-100", "RF-500"):
        n = int(name.split("-")[1])
        return (RandomForestClassifier if clf else RandomForestRegressor)(
            n_estimators=n, min_samples_leaf=2, n_jobs=N_JOBS, random_state=seed)
    if name == "HGB-100":
        return (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(
            max_iter=100, random_state=seed)
    if name == "HGB-500":
        return (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=63, random_state=seed)
    if name in ("MLP-S", "MLP-L"):
        hidden = (64,) if name == "MLP-S" else (256, 256, 128)
        M = MLPClassifier if clf else MLPRegressor
        return make_pipeline(StandardScaler(), M(hidden_layer_sizes=hidden, early_stopping=True,
                                                 max_iter=300, random_state=seed))
    if name == "Stack":
        base = [("rf", build("RF-100", task, seed)), ("hgb", build("HGB-100", task, seed)),
                ("mlp", build("MLP-S", task, seed))]
        if clf:
            return StackingClassifier(base, LogisticRegression(max_iter=2000), cv=3, n_jobs=1)
        return StackingRegressor(base, Ridge(), cv=3, n_jobs=1)
    if name == "HGB-HPO":
        est = (HistGradientBoostingClassifier if clf else HistGradientBoostingRegressor)(random_state=seed)
        space = dict(learning_rate=loguniform(0.01, 0.3), max_iter=randint(100, 600),
                     max_leaf_nodes=randint(15, 127), min_samples_leaf=randint(10, 200),
                     l2_regularization=uniform(0, 1.0))
        return RandomizedSearchCV(est, space, n_iter=20, cv=3, random_state=seed, n_jobs=1,
                                  scoring="roc_auc" if clf else "neg_mean_absolute_error")
    raise ValueError(name)
