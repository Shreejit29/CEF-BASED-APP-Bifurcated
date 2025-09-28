import os, io, json, argparse
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib, shap
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier

TARGET_W = {"cef_slope":0.364803, "cef_std":0.286279, "cef_range":0.216227, "cef_var":0.132692}

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
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing columns after rename: {miss}")
    if "cef_var" not in df.columns:
        df["cef_var"] = df["cef_std"]**2
    df = df.dropna(subset=["cef_slope","cef_range","cef_std","cef_var","label"]).reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return df

def get_pipeline_model(n_estimators=200, random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("gbc", GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state))
    ])

def group_stratified_split(labels, groups, test_size=0.25, random_state=42):
    rng = np.random.RandomState(random_state)
    group_to_label = {}
    for g, y in zip(groups, labels):
        group_to_label[g] = int(y)
    by_label = {}
    for g, lbl in group_to_label.items():
        by_label.setdefault(lbl, []).append(g)
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

def plot_roc(y_true, y_score, out_path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(5,4))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0,1],[0,1],"k--",alpha=0.5)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC (Test)"); plt.legend(loc="lower right")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def plot_pr(y_true, y_score, out_path):
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    plt.figure(figsize=(5,4))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("PR (Test)"); plt.legend(loc="lower left")
    plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

def compute_and_save_shap(model, X_train, feature_names, output_dir):
    scaler = model.named_steps.get("scaler", None)
    clf = model.named_steps.get("gbc", None)
    X_scaled = scaler.transform(X_train) if scaler is not None else X_train

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_scaled)
    shap_values = sv[1] if isinstance(sv, list) and len(sv)>=2 else sv

    shap_summary_png = os.path.join(output_dir, "shap_summary.png")
    shap_bar_png = os.path.join(output_dir, "shap_bar.png")

    plt.figure(); shap.summary_plot(shap_values, features=X_scaled, feature_names=feature_names, show=False)
    plt.tight_layout(); plt.savefig(shap_summary_png, dpi=200); plt.close()

    plt.figure(); shap.summary_plot(shap_values, features=X_scaled, feature_names=feature_names, plot_type="bar", show=False)
    plt.tight_layout(); plt.savefig(shap_bar_png, dpi=200); plt.close()

    mean_abs = np.mean(np.abs(shap_values), axis=0)
    shap_importance_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
    return shap_importance_df, shap_summary_png, shap_bar_png

def build_model_report(data_path, sheet=0, model_path=None, output_dir="model_report", test_size=0.25, random_state=42, n_estimators=200, slope_weight=1.0, k=5.0):
    os.makedirs(output_dir, exist_ok=True)
    df = load_excel_features(data_path, sheet=sheet)
    feature_names = ["cef_slope","cef_range","cef_std","cef_var"]
    X = df[feature_names].values.astype(float)
    y = df["label"].values.astype(int)
    groups = df["cell_id"].values

    # Apply identical multipliers as the app
    mul = np.array([
        (1.0 + k * TARGET_W["cef_slope"]) * float(slope_weight),
        (1.0 + k * TARGET_W["cef_range"]),
        (1.0 + k * TARGET_W["cef_std"]),
        (1.0 + k * TARGET_W["cef_var"]),
    ], dtype=float)
    X = X * mul

    train_idx, test_idx = group_stratified_split(y, groups, test_size=test_size, random_state=random_state)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    if model_path and os.path.exists(model_path):
        model = joblib.load(model_path)
        trained_from = "loaded"
    else:
        model = Pipeline([("scaler", StandardScaler()), ("gbc", GradientBoostingClassifier(n_estimators=n_estimators, random_state=random_state))])
        model.fit(X_train, y_train)
        trained_from = "trained"
        joblib.dump(model, os.path.join(output_dir, "cef_gb_model.joblib"))

    shap_importance_df, shap_summary_png, shap_bar_png = compute_and_save_shap(model, X_train, feature_names, output_dir)

    y_pred = model.predict(X_test)
    y_proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_proba = proba[:, 1]

    report_txt = classification_report(y_test, y_pred, digits=4, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    roc_auc = float(roc_auc_score(y_test, y_proba)) if y_proba is not None and len(np.unique(y_test))==2 else None
    ap = float(average_precision_score(y_test, y_proba)) if y_proba is not None and len(np.unique(y_test))==2 else None

    if y_proba is not None and len(np.unique(y_test))==2:
        plot_roc(y_test, y_proba, os.path.join(output_dir, "roc_curve.png"))
        plot_pr(y_test, y_proba, os.path.join(output_dir, "pr_curve.png"))

    gkf = GroupKFold(n_splits=5)
    cv_scores = cross_val_score(model, X_train, y_train, scoring="f1", cv=gkf.split(X_train, y_train, groups=groups_train), n_jobs=-1)

    xls_path = os.path.join(output_dir, "model_performance_report.xlsx")
    summary = {
        "data_path": data_path, "sheet": sheet,
        "model_source": trained_from if model_path is None else f"loaded({model_path})",
        "n_samples": int(df.shape[0]), "n_train": int(len(train_idx)), "n_test": int(len(test_idx)),
        "cv_f1_mean": float(cv_scores.mean()), "cv_f1_std": float(cv_scores.std()),
        "roc_auc": roc_auc, "average_precision": ap,
        "slope_weight": float(slope_weight), "k": float(k)
    }

    test_df = pd.DataFrame({
        "cell_id": groups[test_idx],
        "cef_slope": X_test[:,0], "cef_range": X_test[:,1], "cef_std": X_test[:,2], "cef_var": X_test[:,3],
        "y_true": y_test, "y_pred": y_pred,
        "y_proba": y_proba if y_proba is not None else [np.nan]*len(y_pred)
    })
    cm_df = pd.DataFrame(cm, index=["True_0","True_1"], columns=["Pred_0","Pred_1"])

    with pd.ExcelWriter(xls_path, engine="xlsxwriter") as writer:
        pd.DataFrame([summary]).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame({"classification_report":[report_txt]}).to_excel(writer, sheet_name="ClassificationReport", index=False)
        cm_df.to_excel(writer, sheet_name="ConfusionMatrix")
        pd.DataFrame({"CV_F1_Scores": cv_scores}).to_excel(writer, sheet_name="CV_Scores", index=False)
        test_df.to_excel(writer, sheet_name="Test_Predictions", index=False)
        if shap_importance_df is not None:
            shap_importance_df.to_excel(writer, sheet_name="SHAP_Importance", index=False)

    return {
        "report_excel": xls_path,
        "roc_png": os.path.join(output_dir, "roc_curve.png") if roc_auc is not None else None,
        "pr_png": os.path.join(output_dir, "pr_curve.png") if ap is not None else None,
        "shap_summary_png": shap_summary_png, "shap_bar_png": shap_bar_png,
        "model_path": os.path.join(output_dir, "cef_gb_model.joblib") if trained_from=="trained" else model_path,
        "summary": summary, "classification_report": report_txt,
        "confusion_matrix": cm.tolist()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model performance report for 10-cycle CEF Gradient Boosting with SHAP and target importance multipliers.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--sheet", default=0)
    parser.add_argument("--model", default=None)
    parser.add_argument("--outdir", default="model_report")
    parser.add_argument("--test_size", type=float, default=0.25)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--slope_weight", type=float, default=1.0)
    parser.add_argument("--k", type=float, default=5.0)
    args = parser.parse_args()

    res = build_model_report(
        data_path=args.data, sheet=args.sheet, model_path=args.model, output_dir=args.outdir,
        test_size=args.test_size, random_state=args.random_state, n_estimators=args.n_estimators,
        slope_weight=args.slope_weight, k=args.k
    )
    print("Report Excel:", res["report_excel"])
    if res["roc_png"]: print("ROC PNG:", res["roc_png"])
    if res["pr_png"]: print("PR PNG:", res["pr_png"])
    if res["shap_summary_png"]: print("SHAP Summary PNG:", res["shap_summary_png"])
    if res["shap_bar_png"]: print("SHAP Bar PNG:", res["shap_bar_png"])
    print("Model path:", res["model_path"])
    print("Summary:", json.dumps(res["summary"], indent=2))
