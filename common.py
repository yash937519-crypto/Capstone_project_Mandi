"""Shared helpers (importing this file never runs a Part)."""
import csv
import os
import numpy as np

OUT = "outputs"


def load_data(path=os.path.join("data", "records.csv")):
    """Return X (800x7 float), y (outcome, int), value (float)."""
    with open(path, newline="") as f:
        rows = list(csv.reader(f))[1:]
    X = np.array([[float(v) for v in r[1:8]] for r in rows])
    y = np.array([int(r[8]) for r in rows])
    value = np.array([float(r[9]) for r in rows])
    return X, y, value


def fmt(v):
    if isinstance(v, (bool, np.bool_)):
        return "yes" if v else "no"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    if isinstance(v, str):
        return v
    return f"{float(v):.4f}"


def write_results(name, d):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w") as f:
        for k, v in d.items():
            f.write(f"{k} = {fmt(v)}\n")


def write_csv(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, name), "w", newline="") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(fmt(v) for v in r) + "\n")


def sample_sd(a, axis=None):
    return np.std(a, axis=axis, ddof=1)


# ---------- Part 2 shared pieces ----------
def stratified_split(y):
    """Task 2.1: first floor(7*count/10) of each class (file order) -> train."""
    train = []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        k = (7 * len(idx)) // 10
        train.extend(idx[:k].tolist())
    train = np.array(sorted(train))
    test = np.array([i for i in range(len(y)) if i not in set(train.tolist())])
    return train, test


def standardise_train_test(X, train, test):
    mu = X[train].mean(axis=0)
    sd = sample_sd(X[train], axis=0)
    return (X[train] - mu) / sd, (X[test] - mu) / sd


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def bce(y, p):
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def train_logistic(Xs, y, eta=0.1, updates=1000, record=()):
    n, d = Xs.shape
    w = np.zeros(d)
    b = 0.0
    losses = {}
    if 0 in record:
        losses[0] = bce(y, sigmoid(Xs @ w + b))
    for t in range(1, updates + 1):
        p = sigmoid(Xs @ w + b)
        err = p - y
        w = w - eta * (Xs.T @ err) / n
        b = b - eta * err.mean()
        if t in record:
            losses[t] = bce(y, sigmoid(Xs @ w + b))
    return w, b, losses


def confusion(y, pred):
    tp = int(np.sum((pred == 1) & (y == 1)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    return tp, fp, tn, fn


def safe_div(a, b):
    return a / b if b != 0 else 0.0


def metrics(tp, fp, tn, fn):
    acc = safe_div(tp + tn, tp + fp + tn + fn)
    prec = safe_div(tp, tp + fp)
    rec = safe_div(tp, tp + fn)
    f1 = safe_div(2 * prec * rec, prec + rec)
    spec = safe_div(tn, tn + fp)
    return acc, prec, rec, f1, spec


def knn_predict(ref_X, ref_y, query_X, K):
    """Return (predicted label, positive share) for every query row."""
    labels, shares = [], []
    for q in query_X:
        d = np.sqrt(((ref_X - q) ** 2).sum(axis=1))
        order = np.argsort(d, kind="stable")[:K]
        share = ref_y[order].sum() / K
        shares.append(share)
        labels.append(1 if ref_y[order].sum() >= (K + 1) / 2 else 0)
    return np.array(labels), np.array(shares)
