"""Part 4 - model-choice and architecture memo."""
import math
import numpy as np
from common import (load_data, write_results, write_csv, stratified_split, standardise_train_test,
                    sigmoid, train_logistic, confusion, metrics, knn_predict)

X, y, value = load_data()
tr, te = stratified_split(y)
Xtr_s, Xte_s = standardise_train_test(X, tr, te)
ytr, yte = y[tr], y[te]
Xtr_raw, Xte_raw = X[tr], X[te]          # original units for the splits
R = {}


# 4.1 impurity and splits
def entropy(pos, n):
    if n == 0:
        return 0.0
    e = 0.0
    for c in (pos / n, 1 - pos / n):
        if c > 0:
            e -= c * math.log2(c)
    return e


def gini(pos, n):
    if n == 0:
        return 0.0
    p = pos / n
    return 1 - p * p - (1 - p) ** 2


N, P = len(ytr), int(ytr.sum())
R["root_entropy"], R["root_gini"] = entropy(P, N), gini(P, N)
splits = {"S1": lambda A: A[:, 5] == 1, "S2": lambda A: A[:, 6] == 1,
          "S3": lambda A: A[:, 2] > 0.5, "S4": lambda A: A[:, 0] <= 24}
rows, ig, gg = [], {}, {}
for nm, f in splits.items():
    m = f(Xtr_raw)
    ny, py = int(m.sum()), int(ytr[m].sum())
    nn, pn = N - ny, P - py
    ig[nm] = entropy(P, N) - (ny / N * entropy(py, ny) + nn / N * entropy(pn, nn))
    gg[nm] = gini(P, N) - (ny / N * gini(py, ny) + nn / N * gini(pn, nn))
    rows.append([nm, ny, py, nn, pn, ig[nm], gg[nm]])
write_csv("part4_splits.csv", ["split", "n_yes", "positives_yes", "n_no", "positives_no",
                               "info_gain", "gini_gain"], rows)
best_ig = max(ig, key=ig.get)
R["best_split_by_info_gain"] = best_ig
R["best_split_by_gini_gain"] = max(gg, key=gg.get)

# 4.2 stump + five predictions on the test rows
m_tr = splits[best_ig](Xtr_raw)
rate_yes, rate_no = ytr[m_tr].mean(), ytr[~m_tr].mean()
R["stump_rate_yes"], R["stump_rate_no"] = rate_yes, rate_no
m_te = splits[best_ig](Xte_raw)
p_stump = np.where(m_te, rate_yes, rate_no)

w, b, _ = train_logistic(Xtr_s, ytr, eta=0.1, updates=1000)
p_log = sigmoid(Xte_s @ w + b)

# KNN: best K comes from Part 2 CV (recomputed here with the shared helper)
Ks = list(range(1, 22, 2))
folds = np.arange(len(tr)) % 5
best_k, best_f1 = None, -1
for K in Ks:
    f1s = []
    for f in range(5):
        ref, hold = folds != f, folds == f
        pred, _ = knn_predict(Xtr_s[ref], ytr[ref], Xtr_s[hold], K)
        f1s.append(metrics(*confusion(ytr[hold], pred))[3])
    if np.mean(f1s) > best_f1 + 1e-12:
        best_k, best_f1 = K, float(np.mean(f1s))
knn_lab, p_knn = knn_predict(Xtr_s, ytr, Xte_s, best_k)

pred = {"logistic": (p_log >= 0.3).astype(int),
        "knn": knn_lab,
        "stump": (p_stump >= 0.3).astype(int)}
votes = pred["logistic"] + pred["knn"] + pred["stump"]
pred["hard_vote"] = (votes >= 2).astype(int)
pred["soft_vote"] = (((p_log + p_knn + p_stump) / 3) >= 0.3).astype(int)
vrows = []
for nm in ("logistic", "knn", "stump", "hard_vote", "soft_vote"):
    tp, fp, tn, fn = confusion(yte, pred[nm])
    _, prec, rec, f1, _ = metrics(tp, fp, tn, fn)
    vrows.append([nm, tp, fp, tn, fn, prec, rec, f1])
write_csv("part4_voting.csv", ["model", "tp", "fp", "tn", "fn", "precision", "recall", "f1"], vrows)

# 4.3 two rounds of AdaBoost by hand
yy = np.array([1, 1, 1, -1, -1, -1, 1, -1])
h1 = np.array([1, 1, -1, -1, -1, -1, -1, -1])
h2 = np.array([-1, 1, 1, -1, 1, -1, 1, -1])
wts = np.full(8, 1 / 8)
alphas, arows = [], []
for rnd, h in ((1, h1), (2, h2)):
    eps = wts[h != yy].sum()
    alpha = 0.5 * math.log((1 - eps) / eps)
    unn = wts * np.exp(-alpha * yy * h)
    new = unn / unn.sum()
    for i in range(8):
        arows.append([rnd, i + 1, wts[i], unn[i], new[i]])
    R[f"round{rnd}_error"], R[f"round{rnd}_alpha"] = eps, alpha
    alphas.append(alpha)
    wts = new
write_csv("part4_adaboost.csv", ["round", "point", "weight_before", "weight_unnormalised",
                                 "weight_after"], arows)
ens = np.sign(alphas[0] * h1 + alphas[1] * h2)
R["ensemble_accuracy"] = float(np.mean(ens == yy))

# 4.4 out-of-bag fraction
R["oob_fraction"] = (1 - 1 / 560) ** 560


# 4.5 architectures (parameters only; nothing is trained)
def conv_out(i, f, pad, s):
    return (i - f + 2 * pad) // s + 1


mlp = [("mlp_hidden1", 784 * 16 + 16), ("mlp_hidden2", 16 * 16 + 16), ("mlp_output", 16 * 10 + 10)]
s1 = conv_out(28, 5, 2, 1)          # 28
p1 = conv_out(s1, 2, 0, 2)          # 14
s2 = conv_out(p1, 5, 0, 1)          # 10
p2 = conv_out(s2, 2, 0, 2)          # 5
flat = p2 * p2 * 16                 # 400
cnn = [("cnn_conv1", 5 * 5 * 1 * 6 + 6), ("cnn_conv2", 5 * 5 * 6 * 16 + 16),
       ("cnn_dense1", flat * 32 + 32), ("cnn_dense2", 32 * 10 + 10)]
write_csv("part4_architecture.csv", ["layer", "parameters"], [[k, v] for k, v in mlp + cnn])
R["mlp_total"] = sum(v for _, v in mlp)
R["cnn_total"] = sum(v for _, v in cnn)
R["fc_layer_params"] = 128 * 128 * 3 * 1024 + 1024
R["conv_layer_params"] = 3 * 3 * 3 * 64 + 64
R["rnn_total"] = 128 * 64 + 128 * 128 + 16 * 128 + 128 + 16

write_results("part4_results.txt", R)
print("cnn shapes:", s1, p1, s2, p2, flat, "| knn K:", best_k)
