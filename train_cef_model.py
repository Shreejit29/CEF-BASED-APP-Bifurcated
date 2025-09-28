import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os

def load_excel_features(file_like_or_path, sheet=0):
    df_raw = pd.read_excel(file_like_or_path, sheet_name=sheet)

    col_map = {
        "Cell ID":"cell_id","cell_id":"cell_id","cell":"cell_id",
        "CEF slope":"cef_slope","cef_slope":"cef_slope",
        "CEF range":"cef_range","cef_range":"cef_range",
        "CEF std":"cef_std","cef_std":"cef_std",
        "CEF variance":"cef_var","cef_variance":"cef_var","cef_var":"cef_var",
        "Label":"label","label":"label","target":"label"
    }
    df = df_raw.rename(columns={k:v for k,v in col_map.items() if k in df_raw.columns})

    required = {"cell_id","cef_slope","cef_range","cef_std","label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in Excel after rename: {missing}")

    if "cef_var" not in df.columns:
        df["cef_var"] = df["cef_std"]**2

    df = df.dropna(subset=["cef_slope","cef_range","cef_std","cef_var","label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df

def group_stratified_split(labels, groups, test_size=0.2, random_state=42):
    rng = np.random.RandomState(random_state)
    group_to_label = {}
    for g, y in zip(groups, labels):
        group_to_label[g] = int(y)
    by_label = defaultdict(list)
    for g, lbl in group_to_label.items():
        by_label[lbl].append(g)
    train_groups, test_groups = [], []
    for lbl, glist in by_label.items():
        glist = np.array(glist)
        rng.shuffle(glist)
        n_test = max(1, int(round(test_size * len(glist))))
        test_groups.extend(glist[:n_test])
        train_groups.extend(glist[n_test:])
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    return np.where(train_mask)[0], np.where(test_mask)[0]

def train_from_dataframe(df, random_state=42, slope_weight=3.0):
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

    feature_names = ["cef_slope","cef_range","cef_std","cef_var"]
    X = df[feature_names].values.astype(float)
    y = df["label"].values.astype(int)
    groups_all = df["cell_id"].values

    # Reweight slope column (index 0)
    X[:, 0] *= float(slope_weight)

    # Split
    train_idx, test_idx = group_stratified_split(y, groups_all, test_size=0.2, random_state=random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups_all[train_idx]

    # Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('gbc', GradientBoostingClassifier(n_estimators=200, random_state=random_state))
    ])

    pipeline.fit(X_train, y_train)

    # Evaluation
    y_pred = pipeline.predict(X_test)
    proba_method = getattr(pipeline, "predict_proba", None)
    y_proba_scores = None
    if callable(proba_method):
        y_proba = proba_method(X_test)
        if isinstance(y_proba, np.ndarray) and y_proba.ndim == 2 and y_proba.shape[1] >= 2:
            y_proba_scores = y_proba[:, 1]
        elif isinstance(y_proba, np.ndarray) and y_proba.ndim == 1:
            y_proba_scores = y_proba

    report = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    auc = None
    if y_proba_scores is not None and len(np.unique(y_test)) == 2:
        auc = float(roc_auc_score(y_test, y_proba_scores))

    # Cross-val
    gkf = GroupKFold(n_splits=5)
    cv_iter = list(gkf.split(X_train, y_train, groups=groups_train))
    cv_scores = cross_val_score(pipeline, X_train, y_train, scoring="f1", cv=cv_iter, n_jobs=-1)

    return {
        "best_params": {"n_estimators": 200, "random_state": random_state, "slope_weight": float(slope_weight)},
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "test_report": report,
        "test_confusion_matrix": cm,
        "test_auc": auc,
        "model": pipeline
    }

def save_model(model, path="models/cef_gb_model.joblib"):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    joblib.dump(model, path)
    return path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Excel path (.xlsx/.xls) with cell_id, cef_slope, cef_range, cef_std, [cef_var], label")
    parser.add_argument("--sheet", default=0, help="Sheet index or name")
    parser.add_argument("--out", default="models/cef_gb_model.joblib")
    args = parser.parse_args()

    df = load_excel_features(args.data, sheet=args.sheet)
    res = train_from_dataframe(df, random_state=42)
    print("Best params:", res["best_params"])
    print("CV F1 mean/std:", res["cv_f1_mean"], res["cv_f1_std"])
    print("Test report:\n", res["test_report"])
    print("Test AUC:", res["test_auc"])
    print("Confusion matrix:", res["test_confusion_matrix"])
    path = save_model(res["model"], args.out)
    print("Saved model to:", path)
