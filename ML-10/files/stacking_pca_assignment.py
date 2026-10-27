
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

RANDOM_STATE = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)

# %% [markdown]
# ## Load data and train/test split

# %%
data = load_breast_cancer(as_frame=True)
X, y = data.data, data.target
print("Shape:", X.shape)
print("Class distribution:\n", y.value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)


# %%
def get_base_models():
    return [
        ("lr",  Pipeline([("sc", StandardScaler()), ("m", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))])),
        ("knn", Pipeline([("sc", StandardScaler()), ("m", KNeighborsClassifier(n_neighbors=5))])),
        ("svm", Pipeline([("sc", StandardScaler()), ("m", SVC(probability=True, random_state=RANDOM_STATE))])),
        ("dt",  DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("nb",  GaussianNB()),
    ]

def evaluate(model, Xtr, Xte, ytr, yte):
    model.fit(Xtr, ytr)
    p = model.predict(Xte)
    return {
        "Accuracy":  accuracy_score(yte, p),
        "Precision": precision_score(yte, p),
        "Recall":    recall_score(yte, p),
        "F1":        f1_score(yte, p),
    }
base_results = {name: evaluate(m, X_train, X_test, y_train, y_test) for name, m in get_base_models()}
base_df = pd.DataFrame(base_results).T
print("Base model performance:")
display_df = base_df.round(4)
print(display_df)

meta_models = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "DecisionTree":       DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE),
    "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "GradientBoosting":   GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVC":                SVC(random_state=RANDOM_STATE),
    "KNN":                KNeighborsClassifier(n_neighbors=5),
}

stack_results = {}
for name, meta in meta_models.items():
    stack = StackingClassifier(
        estimators=get_base_models(),
        final_estimator=meta,
        cv=5,
        n_jobs=-1,
    )
    stack_results[f"Stacking ({name})"] = evaluate(stack, X_train, X_test, y_train, y_test)

stack_df = pd.DataFrame(stack_results).T
print("Stacking results with different meta-models:")
print(stack_df.round(4))

all_df = pd.concat([base_df, stack_df]).sort_values("Accuracy", ascending=False)
print(all_df.round(4))

plt.figure(figsize=(10, 5))
all_df["Accuracy"].sort_values().plot(kind="barh", color="steelblue")
plt.xlim(0.9, 1.0)
plt.xlabel("Test Accuracy")
plt.title("A1: Base models vs Stacking classifiers")
plt.tight_layout()
plt.savefig("A1_stacking_comparison.png", dpi=150)
plt.show()

best_meta = stack_df["Accuracy"].idxmax()
print("Best stacking model:", best_meta, "->", round(stack_df["Accuracy"].max(), 4))


corr = X.corr()

plt.figure(figsize=(16, 13))
sns.heatmap(corr, cmap="coolwarm", center=0, square=True, linewidths=0.2,
            cbar_kws={"shrink": 0.7})
plt.title("A2: Feature correlation heatmap")
plt.tight_layout()
plt.savefig("A2_correlation_heatmap.png", dpi=150)
plt.show()

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
pairs = (upper.stack().rename("correlation").reset_index()
         .rename(columns={"level_0": "feature_1", "level_1": "feature_2"}))
high = pairs[pairs["correlation"].abs() > 0.9].sort_values("correlation", ascending=False)
print(f"Number of feature pairs with |r| > 0.9: {len(high)}")
print(high.head(15).to_string(index=False))

# Correlation of each feature with the target
target_corr = X.corrwith(y).sort_values()
plt.figure(figsize=(8, 8))
target_corr.plot(kind="barh", color="teal")
plt.title("Correlation of features with target")
plt.tight_layout()
plt.savefig("A2_target_correlation.png", dpi=150)
plt.show()


# ## A3. PCA with 99% explained variance

# %%
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# Full PCA to inspect cumulative variance
pca_full = PCA().fit(X_train_sc)
cumvar = np.cumsum(pca_full.explained_variance_ratio_)
n_components = int(np.argmax(cumvar >= 0.99) + 1)
print(f"Components needed for 99% variance: {n_components} (out of {X.shape[1]})")

plt.figure(figsize=(8, 5))
plt.plot(range(1, len(cumvar) + 1), cumvar, marker="o")
plt.axhline(0.99, color="r", linestyle="--", label="99% variance")
plt.axvline(n_components, color="g", linestyle="--", label=f"{n_components} components")
plt.xlabel("Number of components")
plt.ylabel("Cumulative explained variance")
plt.title("A3: PCA cumulative explained variance")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("A3_pca_cumulative_variance.png", dpi=150)
plt.show()

# %%
pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)
print("Original shape:", X_train_sc.shape, "-> PCA shape:", X_train_pca.shape)

# %%
# Models on PCA features (data already scaled, so use plain models)
def get_models_pca():
    return [
        ("lr",  LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("svm", SVC(probability=True, random_state=RANDOM_STATE)),
        ("dt",  DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE)),
        ("rf",  RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)),
        ("nb",  GaussianNB()),
    ]

pca_results = {name: evaluate(m, X_train_pca, X_test_pca, y_train, y_test) for name, m in get_models_pca()}

# Stacking on PCA features (best meta-model from A1 type: use LogisticRegression & RandomForest)
for meta_name in ["LogisticRegression", "RandomForest"]:
    st = StackingClassifier(estimators=get_models_pca(), final_estimator=meta_models[meta_name], cv=5, n_jobs=-1)
    pca_results[f"Stacking ({meta_name})"] = evaluate(st, X_train_pca, X_test_pca, y_train, y_test)

pca_df = pd.DataFrame(pca_results).T

# %%
# Comparison: original vs PCA
orig_df = pd.concat([base_df, stack_df.loc[["Stacking (LogisticRegression)", "Stacking (RandomForest)"]]])
comparison = pd.DataFrame({
    "Accuracy (Original)": orig_df["Accuracy"],
    "Accuracy (PCA)":      pca_df["Accuracy"],
    "F1 (Original)":       orig_df["F1"],
    "F1 (PCA)":            pca_df["F1"],
})
comparison["Acc Diff (PCA - Orig)"] = comparison["Accuracy (PCA)"] - comparison["Accuracy (Original)"]
print(f"Features: {X.shape[1]} (original) vs {n_components} (PCA)")
print(comparison.round(4))

ax = comparison[["Accuracy (Original)", "Accuracy (PCA)"]].plot(kind="bar", figsize=(11, 5), rot=30)
ax.set_ylim(0.9, 1.0)
ax.set_title(f"A3: Original ({X.shape[1]} features) vs PCA ({n_components} components)")
ax.set_ylabel("Test Accuracy")
plt.tight_layout()
plt.savefig("A3_original_vs_pca.png", dpi=150)
plt.show()
