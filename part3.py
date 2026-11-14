"""Part 3 - segmentation and dimensionality reduction (all 800 rows, X1..X5)."""
import numpy as np
from common import load_data, write_results, write_csv, sample_sd

X, y, value = load_data()
X5 = X[:, :5]
n = len(X5)
mu, sd = X5.mean(axis=0), sample_sd(X5, axis=0)
Z = (X5 - mu) / sd
R = {}


# 3.1 K-means with pinned starting rows
def kmeans(Z, K):
    step = len(Z) // K
    cent = Z[[i * step for i in range(K)]].copy()
    labels = None
    passes = 0
    while True:
        d = ((Z[:, None, :] - cent[None, :, :]) ** 2).sum(axis=2)
        new = np.argmin(d, axis=1)          # argmin -> lowest index on ties
        passes += 1
        if labels is not None and np.array_equal(new, labels):
            break
        labels = new
        for k in range(K):
            m = labels == k
            if m.any():
                cent[k] = Z[m].mean(axis=0)
    wcss = float(((Z - cent[labels]) ** 2).sum())
    return labels, cent, wcss, passes


res = {K: kmeans(Z, K) for K in range(2, 7)}
w = {K: res[K][2] for K in res}
for K in range(2, 7):
    R[f"wcss_k{K}"] = w[K]

# 3.2 elbow
elbow = None
for K in (3, 4, 5):
    if (w[K] - w[K + 1]) < 0.5 * (w[K - 1] - w[K]):
        elbow = K
        break
R["elbow_k"] = elbow if elbow is not None else "none"

# 3.3 segments for K = 3, ordered by mean X3
labels3 = res[3][0]
order = sorted(range(3), key=lambda k: X5[labels3 == k, 2].mean())
seg_rows, seg_of = [], np.zeros(n, dtype=int)
for name, k in zip("ABC", order):
    m = labels3 == k
    seg_of[m] = "ABC".index(name)
    R[f"size_segment_{name}"] = int(m.sum())
    seg_rows.append([name, int(m.sum())] + [X5[m, j].mean() for j in range(5)] +
                    [y[m].mean(), value[m].mean()])
write_csv("part3_segments.csv",
          ["segment", "size", "mean_X1", "mean_X2", "mean_X3", "mean_X4", "mean_X5",
           "outcome_rate", "mean_value"], seg_rows)

# 3.4 agglomerative clustering on the supplied table
D = np.array([[0.0, 7.9, 7.4, 2.9, 2.9, 6.1],
              [7.9, 0.0, 1.4, 5.9, 5.3, 2.0],
              [7.4, 1.4, 0.0, 5.8, 4.6, 1.3],
              [2.9, 5.9, 5.8, 0.0, 2.8, 4.5],
              [2.9, 5.3, 4.6, 2.8, 0.0, 3.4],
              [6.1, 2.0, 1.3, 4.5, 3.4, 0.0]])


def link(a, b, rule):
    ds = [D[i, j] for i in a for j in b]
    return min(ds) if rule == "single" else (max(ds) if rule == "complete" else sum(ds) / len(ds))


def name(c):
    return "+".join(f"E{i+1}" for i in sorted(c))


h_rows = []
for rule in ("single", "complete", "average"):
    clusters = [[i] for i in range(6)]
    heights = []
    for step in range(1, 6):
        best = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = link(clusters[i], clusters[j], rule)
                if best is None or d < best[0]:
                    best = (d, i, j)
        d, i, j = best
        a, b = clusters[i], clusters[j]
        if min(b) < min(a):
            a, b = b, a
        h_rows.append([rule, step, name(a), name(b), d])
        heights.append(d)
        clusters = [c for k, c in enumerate(clusters) if k not in (i, j)] + [a + b]
    gaps = [heights[k + 1] - heights[k] for k in range(4)]
    g = int(np.argmax(gaps))                 # gap between merge g+1 and g+2
    R[f"{rule}_clusters_at_cut"] = 6 - (g + 1)
write_csv("part3_hierarchical.csv", ["linkage", "merge_no", "cluster_a", "cluster_b", "height"], h_rows)

# 3.5 PCA by the seven steps
mean_vec = Z.mean(axis=0)                    # (1) ~ zeros after standardising
Xc = Z - mean_vec                            # (2)
S = Xc.T @ Xc / (n - 1)                      # (3)
vals, vecs = np.linalg.eigh(S)               # (4)
idx = np.argsort(vals)[::-1]                 # (5)
vals, vecs = vals[idx], vecs[:, idx]
for j in range(vecs.shape[1]):               # sign convention
    if vecs[np.argmax(np.abs(vecs[:, j])), j] < 0:
        vecs[:, j] = -vecs[:, j]
W = vecs[:, :2]                              # (6)
P = Xc @ W                                   # (7)
tot = vals.sum()
R["eigenvalue_sum"] = tot
for j in range(5):
    R[f"eigenvalue_{j+1}"] = vals[j]
for j in range(5):
    R[f"explained_ratio_{j+1}"] = vals[j] / tot
cum = np.cumsum(vals) / tot
for j in range(5):
    R[f"cumulative_ratio_{j+1}"] = cum[j]
R["components_for_80pct"] = int(np.argmax(cum >= 0.8) + 1)
for pc in (0, 1):
    for j in range(5):
        R[f"pc{pc+1}_loading_X{j+1}"] = vecs[j, pc]
for r in range(3):
    R[f"row{r+1}_pc1"] = P[r, 0]
    R[f"row{r+1}_pc2"] = P[r, 1]
pc1_all = Xc @ vecs[:, 0]
for s, nm in enumerate("ABC"):
    R[f"mean_pc1_segment_{nm}"] = pc1_all[seg_of == s].mean()

write_results("part3_results.txt", R)
print("kmeans passes:", {K: res[K][3] for K in res})
