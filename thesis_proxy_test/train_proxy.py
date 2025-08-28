import argparse
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import joblib
import os

def best_f1_threshold(y_true, y_score):
    ths = np.linspace(0.0, 1.0, 501)
    best = (0.0, 0.0)  # f1, th
    for th in ths:
        y_pred = (y_score >= th).astype(int)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)
        if f1 > best[0]:
            best = (f1, th)
    return best

def main():
    ap = argparse.ArgumentParser("Train proxy classifier on YOLO embeddings")
    ap.add_argument("--data", type=str, required=True, help=".npz from extract_embeddings.py")
    ap.add_argument("--test_size", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="./proxy_models")
    args = ap.parse_args()

    pack = np.load(args.data, allow_pickle=True)
    X = pack["X"].astype(np.float32)
    y = pack["y"].astype(np.int64)
    meta = pack["meta"].item() if "meta" in pack.files else {}

    # flatten in case embeddings are 2D
    X = X.reshape(X.shape[0], -1)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y))
    Xtr, Xte = X[train_idx], X[test_idx]
    ytr, yte = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr)
    Xte_s = scaler.transform(Xte)

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs", n_jobs=None)
    clf.fit(Xtr_s, ytr)

    # metrics
    y_score = clf.predict_proba(Xte_s)[:, 1]
    auc = roc_auc_score(yte, y_score)
    ap = average_precision_score(yte, y_score)
    f1, th = best_f1_threshold(yte, y_score)
    y_pred = (y_score >= th).astype(int)

    cm = confusion_matrix(yte, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("Meta:", meta)
    print(f"Samples: total={len(y)}  train={len(ytr)}  test={len(yte)}  positives={int(y.sum())}")
    print(f"ROC AUC: {auc:.3f}")
    print(f"PR  AP:  {ap:.3f}")
    print(f"Best-F1 threshold: {th:.3f}  F1={f1:.3f}")
    print("Confusion at best-F1 [0,1]:")
    print(f"TN={tn}  FP={fp}  FN={fn}  TP={tp}")

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(dict(scaler=scaler, clf=clf, meta=meta), os.path.join(args.out_dir, "proxy_logreg.joblib"))
    print(f"Saved model to {os.path.join(args.out_dir, 'proxy_logreg.joblib')}")

if __name__ == "__main__":
    main()
