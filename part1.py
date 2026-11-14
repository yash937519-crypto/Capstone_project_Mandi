"""Part 1 - estimation, inference and regression baseline (all 800 rows)."""
import math
import numpy as np
from common import load_data, write_results, sample_sd

X, y, value = load_data()
n = len(value)
R = {}

# 1.1 estimate with an interval
mean_value = value.mean()
sd_value = sample_sd(value)
se_value = sd_value / math.sqrt(n)
R.update(n=n, mean_value=mean_value, sd_value=sd_value, se_value=se_value,
         ci95_low=mean_value - 1.96 * se_value, ci95_high=mean_value + 1.96 * se_value)

# 1.2 test H0: mu = 8
z_stat = (mean_value - 8.0) / se_value
R.update(z_stat=z_stat, reject_h0=abs(z_stat) > 1.96)

# 1.3 sample size for E = 0.5
R["n_required"] = int(math.ceil((1.96 * sd_value / 0.5) ** 2))

# 1.4 closed-form simple regression of value on X4
x4 = X[:, 3]


def simple_ols(x, t):
    xm, tm = x.mean(), t.mean()
    cov = np.sum((x - xm) * (t - tm)) / (len(x) - 1)
    var = np.sum((x - xm) ** 2) / (len(x) - 1)
    slope = cov / var
    return slope, tm - slope * xm


slope, intercept = simple_ols(x4, value)
fit = intercept + slope * x4
ym = value.mean()
tss = np.sum((value - ym) ** 2)
ess = np.sum((fit - ym) ** 2)
rss = np.sum((value - fit) ** 2)
R.update(reg_slope=slope, reg_intercept=intercept, tss=tss, ess=ess, rss=rss,
         tss_equals_ess_plus_rss=abs(tss - ess - rss) < 1e-6 * tss,
         r2=1 - rss / tss, rmse=math.sqrt(rss / n), mae=np.mean(np.abs(value - fit)))

# 1.5 gradient descent on standardised X4
x4_mean, x4_sd = x4.mean(), sample_sd(x4)
z = (x4 - x4_mean) / x4_sd
m = c = 0.0
eta = 0.1
costs = {}


def cost(m, c):
    e = value - (m * z + c)
    return float(np.mean(e ** 2))


costs[0] = cost(m, c)
for t in range(1, 501):
    e = value - (m * z + c)
    gm = -2.0 / n * np.sum(z * e)
    gc = -2.0 / n * np.sum(e)
    m -= eta * gm
    c -= eta * gc
    if t in (1, 10, 100, 500):
        costs[t] = cost(m, c)
gd_slope = m / x4_sd
gd_intercept = c - m * x4_mean / x4_sd
R.update(gd_J0=costs[0], gd_J1=costs[1], gd_J10=costs[10], gd_J100=costs[100], gd_J500=costs[500],
         gd_slope=gd_slope, gd_intercept=gd_intercept, gd_slope_gap=abs(gd_slope - slope))


# 1.6 funnel check and log remedy
def half_sds(fitted, resid):
    order = np.argsort(fitted, kind="stable")
    lo, hi = order[:400], order[400:]
    return sample_sd(resid[lo]), sample_sd(resid[hi])


resid = value - fit
lo, hi = half_sds(fit, resid)
R.update(resid_sd_lower_raw=lo, resid_sd_upper_raw=hi, resid_sd_ratio_raw=hi / lo)

lx, ly = np.log(x4), np.log(value)
ll_slope, ll_int = simple_ols(lx, ly)
ll_fit = ll_int + ll_slope * lx
ll_res = ly - ll_fit
ll_rss = np.sum(ll_res ** 2)
ll_tss = np.sum((ly - ly.mean()) ** 2)
ll_r2 = 1 - ll_rss / ll_tss
lo2, hi2 = half_sds(ll_fit, ll_res)
R.update(loglog_intercept=ll_int, loglog_slope=ll_slope, loglog_r2=ll_r2,
         loglog_rmse=math.sqrt(ll_rss / n), resid_sd_lower_log=lo2, resid_sd_upper_log=hi2,
         resid_sd_ratio_log=hi2 / lo2)

# 1.7 multiple regression by the normal equation
A = np.column_stack([np.ones(n), np.log(x4), X[:, 0], X[:, 1], X[:, 2], X[:, 4], X[:, 5], X[:, 6]])
beta = np.linalg.solve(A.T @ A, A.T @ ly)
mr_res = ly - A @ beta
mr_r2 = 1 - np.sum(mr_res ** 2) / ll_tss
p = 7
for i, b in enumerate(beta):
    R[f"mr_beta_{i}"] = b
R["mr_r2"] = mr_r2
R["mr_adj_r2"] = 1 - (1 - mr_r2) * (n - 1) / (n - p - 1)
R["loglog_adj_r2"] = 1 - (1 - ll_r2) * (n - 1) / (n - 1 - 1)

write_results("part1_results.txt", R)
