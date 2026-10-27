
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import loguniform, uniform

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, ConfusionMatrixDisplay)

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## Load data and split

# %%
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
class_names = [str(c) for c in data.target_names]   # ['malignant', 'benign']
print("Shape:", X.shape, "| classes:", class_names)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# %%
def get_scores(model, Xte, yte):
    """Return all metrics for a fitted model. ROC-AUC uses predict_proba or decision_function."""
    pred = model.predict(Xte)
    if hasattr(model, "predict_proba"):
        score = model.predict_proba(Xte)[:, 1]
    else:
        score = model.decision_function(Xte)
    return {
        "Accuracy":  accuracy_score(yte, pred),
        "Precision": precision_score(yte, pred),
        "Recall":    recall_score(yte, pred),
        "F1":        f1_score(yte, pred),
        "ROC-AUC":   roc_auc_score(yte, score),
    }

# %% [markdown]
# ## A1. Baseline Perceptron and MLP (continuing previous experiments)
# Scaling is required for neural models, so each model sits in a `Pipeline` with `StandardScaler`.

# %%
base_perceptron = Pipeline([("scaler", StandardScaler()),
                            ("clf", Perceptron(random_state=RANDOM_STATE))])
base_mlp = Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=RANDOM_STATE))])

baseline = {}
for name, model in [("Perceptron (baseline)", base_perceptron), ("MLP (baseline)", base_mlp)]:
    model.fit(X_train, y_train)
    baseline[name] = get_scores(model, X_test, y_test)

baseline_df = pd.DataFrame(baseline).T
print(baseline_df.round(4))

# %% [markdown]
# ## A2. Hyperparameter tuning with RandomizedSearchCV

# %%
# --- Perceptron ---
perc_pipe = Pipeline([("scaler", StandardScaler()),
                      ("clf", Perceptron(random_state=RANDOM_STATE))])
perc_params = {
    "clf__penalty":      [None, "l2", "l1", "elasticnet"],
    "clf__alpha":        loguniform(1e-5, 1e-1),
    "clf__eta0":         loguniform(1e-2, 10),
    "clf__max_iter":     [500, 1000, 2000],
    "clf__tol":          loguniform(1e-5, 1e-2),
    "clf__fit_intercept": [True, False],
    "clf__class_weight": [None, "balanced"],
}
perc_search = RandomizedSearchCV(perc_pipe, perc_params, n_iter=40, cv=cv,
                                 scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1)
perc_search.fit(X_train, y_train)
print("Perceptron best params:", perc_search.best_params_)
print("Perceptron best CV accuracy:", round(perc_search.best_score_, 4))

# %%
# --- MLP ---
mlp_pipe = Pipeline([("scaler", StandardScaler()),
                     ("clf", MLPClassifier(max_iter=1000, random_state=RANDOM_STATE))])
mlp_params = {
    "clf__hidden_layer_sizes": [(50,), (100,), (150,), (50, 25), (100, 50), (100, 50, 25)],
    "clf__activation":         ["relu", "tanh", "logistic"],
    "clf__solver":             ["adam", "sgd", "lbfgs"],
    "clf__alpha":              loguniform(1e-5, 1e-1),
    "clf__learning_rate_init": loguniform(1e-4, 1e-1),
    "clf__batch_size":         [16, 32, 64, "auto"],
}
mlp_search = RandomizedSearchCV(mlp_pipe, mlp_params, n_iter=30, cv=cv,
                                scoring="accuracy", random_state=RANDOM_STATE, n_jobs=-1)
mlp_search.fit(X_train, y_train)
print("MLP best params:", mlp_search.best_params_)
print("MLP best CV accuracy:", round(mlp_search.best_score_, 4))

# %%
# Baseline vs tuned (test set)
tuned = {
    "Perceptron (tuned)": get_scores(perc_search.best_estimator_, X_test, y_test),
    "MLP (tuned)":        get_scores(mlp_search.best_estimator_, X_test, y_test),
}
tuned_df = pd.DataFrame(tuned).T
before_after = pd.concat([baseline_df, tuned_df]).loc[
    ["Perceptron (baseline)", "Perceptron (tuned)", "MLP (baseline)", "MLP (tuned)"]]
print(before_after.round(4))

# Top of the search results for reference
cv_res = pd.DataFrame(mlp_search.cv_results_).sort_values("rank_test_score")
print("\nTop 5 MLP configurations:")
print(cv_res[["mean_test_score", "std_test_score", "params"]].head(5).to_string(index=False))

# %% [markdown]
# ## A3. Compare with other classifiers
# Distance/margin based models (SVM, NB) are placed in a scaling pipeline; tree-based models do not need scaling.

# %%
models = {
    "SVM":           Pipeline([("scaler", StandardScaler()), ("clf", SVC(probability=True, random_state=RANDOM_STATE))]),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "CatBoost":      CatBoostClassifier(iterations=300, verbose=0, random_state=RANDOM_STATE),
    "AdaBoost":      AdaBoostClassifier(n_estimators=200, random_state=RANDOM_STATE),
    "XGBoost":       XGBClassifier(n_estimators=200, eval_metric="logloss", random_state=RANDOM_STATE),
    "Naive Bayes":   GaussianNB(),
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    res = get_scores(model, X_test, y_test)
    res["CV Accuracy (5-fold)"] = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy").mean()
    results[name] = res

for name, est in [("Perceptron (tuned)", perc_search.best_estimator_), ("MLP (tuned)", mlp_search.best_estimator_)]:
    res = get_scores(est, X_test, y_test)
    res["CV Accuracy (5-fold)"] = perc_search.best_score_ if "Perc" in name else mlp_search.best_score_
    results[name] = res

results_df = pd.DataFrame(results).T.sort_values("Accuracy", ascending=False)
print(results_df.round(4))
results_df.round(4).to_csv("A3_results_table.csv")

# %%
ax = results_df[["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]].plot(kind="bar", figsize=(12, 5), rot=30)
ax.set_ylim(0.88, 1.01)
ax.set_title("A3: Classifier comparison (test set)")
ax.legend(loc="lower right", ncol=5)
plt.tight_layout()
plt.savefig("A3_classifier_comparison.png", dpi=150)
plt.show()

# %% [markdown]
# ## A4. Pipeline: preprocessing + classification in one object
# Steps: **Imputer -> StandardScaler -> MLP (tuned hyperparameters)**.
# The pipeline accepts raw data, applies every step in order and returns predictions, which prevents data leakage
# during cross-validation.

# %%
best_mlp_params = {k.replace("clf__", ""): v for k, v in mlp_search.best_params_.items()}

final_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
    ("clf",     MLPClassifier(max_iter=1000, random_state=RANDOM_STATE, **best_mlp_params)),
])

final_pipeline.fit(X_train, y_train)
print(final_pipeline)

pipe_scores = get_scores(final_pipeline, X_test, y_test)
print("\nTest metrics:", {k: round(v, 4) for k, v in pipe_scores.items()})
cv_scores = cross_val_score(final_pipeline, X_train, y_train, cv=cv, scoring="accuracy")
print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

ConfusionMatrixDisplay(confusion_matrix(y_test, final_pipeline.predict(X_test)),
                       display_labels=class_names).plot(cmap="Blues")
plt.title("A4: Pipeline confusion matrix (test set)")
plt.tight_layout()
plt.savefig("A4_pipeline_confusion_matrix.png", dpi=150)
plt.show()

# %% [markdown]
# ## A5. LIME explanations for the pipeline
# LIME needs a function that maps **raw** feature values to class probabilities. We pass
# `final_pipeline.predict_proba`, so the imputer and scaler are applied automatically inside the pipeline.

# %%
from lime.lime_tabular import LimeTabularExplainer

explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=list(X.columns),
    class_names=class_names,
    mode="classification",
    discretize_continuous=True,
    random_state=RANDOM_STATE,
)

def explain_and_save(idx_in_test, filename):
    row = X_test.iloc[idx_in_test]
    exp = explainer.explain_instance(row.values, final_pipeline.predict_proba, num_features=10, num_samples=5000)
    probs = final_pipeline.predict_proba(row.values.reshape(1, -1))[0]
    actual = class_names[y_test.iloc[idx_in_test]]
    pred = class_names[int(np.argmax(probs))]
    print("(positive weight = pushes towards BENIGN, negative weight = pushes towards MALIGNANT)")
    print(f"\nTest sample {idx_in_test}: actual = {actual}, predicted = {pred}, "
          f"P({class_names[0]}) = {probs[0]:.3f}, P({class_names[1]}) = {probs[1]:.3f}")
    for feat, w in exp.as_list():
        print(f"  {feat:45s} {w:+.4f}")
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME - test sample {idx_in_test} (actual: {actual}, predicted: {pred})")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.show()
    exp.save_to_file(filename.replace(".png", ".html"))
    return exp

# %%
# Pick one malignant case, one benign case and (if any) a misclassified case
pred_test = final_pipeline.predict(X_test)
idx_malignant = int(np.where(y_test.values == 0)[0][0])
idx_benign    = int(np.where(y_test.values == 1)[0][0])
wrong = np.where(pred_test != y_test.values)[0]

explain_and_save(idx_malignant, "A5_lime_malignant.png")
explain_and_save(idx_benign, "A5_lime_benign.png")
if len(wrong):
    explain_and_save(int(wrong[0]), "A5_lime_misclassified.png")
else:
    print("No misclassified test samples for this pipeline.")

# %%
# Global view: average absolute LIME weight over 30 test samples
agg = {}
for i in range(min(30, len(X_test))):
    exp = explainer.explain_instance(X_test.iloc[i].values, final_pipeline.predict_proba,
                                     num_features=len(X.columns), num_samples=2000)
    for feat_idx, w in exp.as_map()[1]:
        agg.setdefault(X.columns[feat_idx], []).append(abs(w))

global_imp = pd.Series({k: np.mean(v) for k, v in agg.items()}).sort_values(ascending=False).head(10)
print("\nTop 10 features by mean |LIME weight|:")
print(global_imp.round(4))

plt.figure(figsize=(8, 5))
global_imp.sort_values().plot(kind="barh", color="darkorange")
plt.xlabel("Mean |LIME weight|")
plt.title("A5: Most influential features (30 test samples)")
plt.tight_layout()
plt.savefig("A5_lime_global_importance.png", dpi=150)
plt.show()
