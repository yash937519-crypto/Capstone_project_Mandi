"""Part 2 - classification and evaluation."""
import math
import numpy as np
from common import (load_data, write_results, write_csv, stratified_split, standardise_train_test,
                    sigmoid, bce, train_logistic, confusion, metrics, knn_predict)

X, y, value = load_data()
R = {}

# 2.1 split + standardise (training statistics only)
tr, te = stratified_split(y)
Xtr, Xte = standardise_train_test(X, tr, te)
ytr, yte = y[tr], y[te]
R.update(train_rows=len(tr), train_positives=int(ytr.sum()), test_rows=len(te), test_positives=int(yte.sum()))

# 2.2 logistic regression from scratch
w, b, L = train_logistic(Xtr, ytr, eta=0.1, updates=1000, record=(0, 1, 10, 100, 1000))
for k in (0, 1, 10, 100, 1000):
    R[f"loss_epoch_{k}"] = L[k]
for j in range(7):
    R[f"weight_X{j+1}"] = w[j]
R["bias"] = b
p_te = sigmoid(Xte @ w + b)
p_tr = sigmoid(Xtr @ w + b)
R["test_bce"] = bce(yte, p_te)

# 2.3 learning-rate effect
for eta in (0.01, 0.1, 1.0):
    _, _, l = train_logistic(Xtr, ytr, eta=eta, updates=200, record=(200,))
    R[f"loss_epoch_200_eta_{eta}"] = l[200]

# 2.4 odds ratios
for j in range(7):
    R[f"odds_ratio_X{j+1}"] = math.exp(w[j])

# 2.5 two thresholds
for tau in (0.5, 0.3):
    tp, fp, tn, fn = confusion(yte, (p_te >= tau).astype(int))
    acc, prec, rec, f1, spec = metrics(tp, fp, tn, fn)
    for name, v in zip(("tp", "fp", "tn", "fn", "accuracy", "precision", "recall", "f1", "specificity"),
                       (tp, fp, tn, fn, acc, prec, rec, f1, spec)):
        R[f"tau_{tau}_{name}"] = v
R["blind_accuracy"] = float(np.mean(yte == 0))

# 2.6 threshold sweep
rows = []
for k in range(1, 10):
    tau = k / 10
    tp, fp, tn, fn = confusion(yte, (p_te >= tau - 1e-12).astype(int))
    rows.append([tau, tp, fp, tn, fn, tp / (tp + fn), fp / (fp + tn)])
write_csv("part2_thresholds.csv", ["tau", "tp", "fp", "tn", "fn", "tpr", "fpr"], rows)

# 2.7 KNN with 5-fold CV on training rows (folds by position mod 5, features from Task 2.1)
Ks = list(range(1, 22, 2))
folds = np.arange(len(tr)) % 5
cv_rows = []
for K in Ks:
    f1s = []
    for f in range(5):
        ref, hold = folds != f, folds == f
        pred, _ = knn_predict(Xtr[ref], ytr[ref], Xtr[hold], K)
        tp, fp, tn, fn = confusion(ytr[hold], pred)
        f1s.append(metrics(tp, fp, tn, fn)[3])
    cv_rows.append([K, float(np.mean(f1s))])
write_csv("part2_knn_cv.csv", ["K", "mean_cv_f1"], cv_rows)
best_k = max(cv_rows, key=lambda r: (r[1], -r[0]))[0]   # ties -> smaller K
R["knn_best_k"] = int(best_k)
kp, _ = knn_predict(Xtr, ytr, Xte, best_k)
tp, fp, tn, fn = confusion(yte, kp)
acc, prec, rec, f1, _ = metrics(tp, fp, tn, fn)
R.update(knn_tp=tp, knn_fp=fp, knn_tn=tn, knn_fn=fn, knn_accuracy=acc, knn_precision=prec,
         knn_recall=rec, knn_f1=f1)


# 2.8 threshold chosen with money (on training rows only)
def cost(yy, pp, tau):
    _, fp, _, fn = confusion(yy, (pp >= tau - 1e-12).astype(int))
    return 50 * fp + 400 * fn


best = None
for k in range(1, 20):
    tau = k / 20
    c = cost(ytr, p_tr, tau)
    if best is None or c < best[1]:
        best = (tau, c)
R["cost_tau"] = best[0]
R["cost_train_at_tau"] = best[1]
R["cost_test_at_tau"] = cost(yte, p_te, best[0])
R["cost_test_at_0.5"] = cost(yte, p_te, 0.5)
R["cost_test_flag_nobody"] = 400 * int(yte.sum())

write_results("part2_results.txt", R)
