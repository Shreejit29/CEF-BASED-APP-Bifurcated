import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
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
def train_from_dataframe(df, random_state=42):
    X = df[["cef_slope","cef_range","cef_std","cef_var"]].values
    y = df["label"].values
    groups_all = df["cell_id"].values

    gss = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups_all))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups_all[train_idx]

    clf = HistGradientBoostingClassifier(random_state=random_state)
    param_grid = {
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [None, 3, 5],
        "max_iter": [100, 200, 400]
    }

    gkf = GroupKFold(n_splits=5)
    cv_iter = list(gkf.split(X_train, y_train, groups=groups_train))  # force materialization

    # Debug guard
    if not cv_iter or not isinstance(cv_iter[0], tuple):
        raise RuntimeError("CV iterator malformed")

    gs = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        scoring="f1",
        cv=cv_iter,
        n_jobs=-1
    )
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    y_pred = best.predict(X_test)
    y_proba = getattr(best, "predict_proba", None)
    auc = roc_auc_score(y_test, y_proba[:,1]) if y_proba is not None else None
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    cv_iter_full = list(gkf.split(X_train, y_train, groups=groups_train))  # materialize again
    cv_scores = cross_val_score(best, X_train, y_train, scoring="f1", cv=cv_iter_full, n_jobs=-1)

    return {
        "best_params": gs.best_params_,
        "cv_f1_mean": float(cv_scores.mean()),
        "cv_f1_std": float(cv_scores.std()),
        "test_report": report,
        "test_confusion_matrix": cm.tolist(),
        "test_auc": float(auc) if auc is not None else None,
        "model": best
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
