Domain: Support

# Swiggy Support Decision Intelligence (synthetic data)

All numbers below are copied from the committed files in `outputs/`. The data is synthetic (see `generate_data.py`); nothing here describes Swiggy's real tickets. Money is in INR.

## How to run

From a clean clone of the `main` branch (Python 3.10 or newer):

```
pip install -r requirements.txt
python generate_data.py      # prints: 800 10 170 7cb48ef89914049d76954530b6482e04
python part1.py
python part2.py
python part3.py
python part4.py
```

Each script reads `data/records.csv` and rewrites its own files in `outputs/`. `common.py` holds helpers shared by the Parts (data loading, the 560/240 split, logistic regression, KNN, metrics); importing it runs nothing.

## Part 1 findings

**R1.1** H0 (mean resolution time = 8.0 minutes) is **rejected**: `z_stat` = 2.8930, which is beyond the critical value 1.96 (two-tailed, alpha = 0.05). A Type I error here would mean concluding that the true average resolution time is not 8 minutes when it actually is, and re-planning staffing for no reason. The gap `mean_value` - 8.0 = 8.9382 - 8.0 = 0.9382 minutes (about 56 seconds, roughly 12% more than the plan). That is statistically clear, and across a large ticket volume it is big enough to matter for staffing, although the data is right-skewed (`sd_value` = 9.1723), so the mean overstates a typical ticket.

**R1.2** `n_required` = 1293 tickets for a margin of error of 0.5 minutes at 95% confidence, against 800 rows available (about 493 more). With 800 rows the margin is 1.96 x 0.3243 = about 0.64 minutes, so the planning team can say the average is between 8.30 and 9.57 minutes (`ci95_low` 8.3026, `ci95_high` 9.5738) but needs a bigger sample for the half-minute precision.

**R1.3** `gd_slope_gap` = 0.0000 (about 3e-16). The squared-error cost is a single convex bowl whose bottom is the least-squares line, and 500 steps with eta = 0.1 on standardised X4 reach it (the cost is already flat from update 100: `gd_J100` = `gd_J500` = 40.3698).

**R1.4** `resid_sd_ratio_raw` = 3.7525 and `resid_sd_ratio_log` = 1.0396. The raw model shows a strong funnel: residual spread in the upper half of fitted values is almost four times the lower half. The log-log model removes it (ratio close to 1).

**R1.5** Predictors with |coefficient| > 0.05 (mechanical rule, excluding the constant): ln(X4) (`mr_beta_1` = 1.1576), X2 (`mr_beta_3` = 0.0940), X3 (`mr_beta_4` = 0.1067) and X6 (`mr_beta_6` = 0.1518). This is not a ranking of importance because the units differ. `mr_adj_r2` = 0.7403 versus `loglog_adj_r2` = 0.6561, so the extra predictors earn their place (adjusted R-squared rises by about 0.084 even after the penalty for seven predictors).

## Part 2 findings

**R2.1** From lowest to highest loss after 200 updates: eta = 1.0 (0.3703), eta = 0.1 (0.3776), eta = 0.01 (0.5091). A learning rate that is too small has barely started learning after 200 updates, while a large one is still stable here and reaches the minimum fastest (it would diverge on a harder problem).

**R2.2** The three largest absolute weights are X3 `sla_consumed_ratio` (0.9013), X1 `customer_tenure_months` (-0.5554) and X2 `earlier_ticket_count_90d` (0.5505). The largest odds ratio is X3 at 2.4629: each extra training SD of SLA already consumed multiplies the odds of escalation by about 2.5, other things equal.

**R2.3** `blind_accuracy` = 0.7875 versus 0.8042 accuracy at tau = 0.5. Predicting "no escalation" for everyone scores almost the same, because only about one ticket in five escalates, so accuracy hides whether the escalations are caught. Lowering tau from 0.5 to 0.3 raised recall from 0.4706 to 0.7451 and lowered precision from 0.5455 to 0.4634 (more escalations caught, more unnecessary reviews).

**R2.4** At tau = 0.3 the logistic model has f1 = 0.5714 and recall = 0.7451, against KNN (K = 5) with f1 = 0.3696 and recall = 0.3333. Logistic regression is clearly better on both.

**R2.5** I would deploy the logistic model at tau = 0.15 (`cost_tau`). On the test rows it costs INR 6050 (`cost_test_at_tau`), against INR 11800 at tau = 0.5 (`cost_test_at_0.5`) and INR 20400 if nobody is flagged (`cost_test_flag_nobody`). The threshold was chosen on training rows so that the test rows remain an honest, untouched check of how it performs on new tickets.

## Part 3 findings

**R3.1** WCSS drops: 2 to 3 = 659.8282, 3 to 4 = 108.0739, 4 to 5 = 171.0575, 5 to 6 = 45.5466. The drop is far the biggest from K = 2 to 3 and then falls sharply, so the elbow rule gives `elbow_k` = 3 (the 3-to-4 drop is under half of the 2-to-3 drop). The drops do not shrink steadily (4 to 5 is larger than 3 to 4), which is expected because the pinned starting rows give one reproducible solution, not necessarily the global optimum.

**R3.2** Segments (from `part3_segments.csv`):
- **A "Steady, early-resolved tickets"** (322 tickets): long tenure (47.5 months), few earlier tickets (1.95), only 22.5% of the SLA used, outcome rate 0.0186, mean resolution 3.60 minutes. Action: keep on the normal self-service/standard queue with no supervisor review.
- **B "Heavy repeat contact, long handling"** (255): 6.07 earlier tickets, highest account value (28.7 thousand INR), mean resolution 17.12 minutes, outcome rate 0.2353. Action: route to experienced agents and budget longer handling time in the staffing plan.
- **C "Late-arriving, near-breach tickets"** (223): 71.2% of the SLA already used on arrival, short tenure (14.4 months), low satisfaction (38.5), **highest outcome rate 0.4664**. Action: put at the front of the queue and have a supervisor look at them at intake.

**R3.3** Single linkage cut gives 4 clusters: {E2, E3, E6}, {E1}, {E4} and {E5}. Complete linkage cut gives 2 clusters: {E1, E4, E5} and {E2, E3, E6} (average linkage gives the same 2). Single linkage only looks at the closest pair, so its merge heights stay small (1.3 up to 3.4) and the biggest gap falls early (1.4 to 2.8). Complete linkage uses the farthest pair, so the last merge happens at 7.9 and the two groups are clearly separated.

**R3.4** PC1 is driven by tenure X1 (+0.5317) and satisfaction X5 (+0.4418) pulling one way and SLA ratio X3 (-0.5187), earlier tickets X2 (-0.4100) and account value X4 (-0.2916) pulling the other, so it is a "settled, long-standing customer versus stressed, waiting customer" axis. The mean vector of the standardised data is zero (to rounding). Three components are needed to keep at least 80% of the variance (cumulative 0.7593 for two, 0.8616 for three). `mean_pc1_segment_A/B/C` = 1.7773 / -0.7831 / -1.6709, so PC1 orders the segments from A to C. A is far from the other two, while B and C are closer on PC1 and need PC2 (account value and earlier tickets) to be told apart.

## Part 4 findings

**R4.1** S4 (tenure <= 24 months) is best by both measures (information gain 0.1252, Gini gain 0.0606) because it divides the tickets into two sides with very different escalation rates (45.7% against 8.9%) while the other splits barely change the mix. A tree grown until every leaf is pure memorises noise in the training rows and so does worse on new tickets. A rule that stops it is a maximum depth (or a minimum number of rows per leaf).

**R4.2** I would ship the logistic model at a cost-chosen threshold. At tau = 0.3 it has f1 = 0.5714 and recall = 0.7451; the stump has f1 = 0.5532 and recall = 0.7647, KNN 0.3696 and 0.3333, hard vote 0.5344 and 0.6863, soft vote 0.5507 and 0.7451. Combining models did not help: neither vote beats the logistic model alone, because KNN is weak and drags the votes down.

**R4.3** After round 1 the two misclassified points (3 and 7) carry 0.2500 + 0.2500 = 0.5 of the total weight. Round 2 trusts h2 more because under the new weights h2 has error 0.1667 (alpha = 0.8047) while h1 would have error 0.5, and h1 earned only alpha = 0.5493 in round 1. The ensemble gets 0.7500 of the eight points right.

**R4.4** Model A option 2 (CNN):

| Layer | Output shape (H x W x D) | Parameters |
|---|---|---|
| Input | 28 x 28 x 1 | 0 |
| Conv 5x5, 6 filters, pad 2 | 28 x 28 x 6 | 156 |
| ReLU + MaxPool 2x2 | 14 x 14 x 6 | 0 |
| Conv 5x5, 16 filters | 10 x 10 x 16 | 2416 |
| ReLU + MaxPool 2x2 | 5 x 5 x 16 | 0 |
| Flatten | 400 | 0 |
| Dense 32 + ReLU | 32 | 12832 |
| Dense 10 | 10 | 330 |

`cnn_total` = 15734, against `mlp_total` = 13002 for the fully connected option. For a 128 x 128 x 3 input, a fully connected layer with 1024 neurons needs `fc_layer_params` = 50332672 parameters, whereas a convolution with 64 filters of 3 x 3 needs only `conv_layer_params` = 1792, because the same small filters are reused at every position.

Model B (RNN): W_xh is 128 x 64, W_hh is 128 x 128 and W_hy is 16 x 128, so `rnn_total` = 26768. It does not grow with longer sequences because the same three weight matrices and biases are reused at every time step.

For 10 classes I would use ReLU in the hidden layers and softmax at the output with categorical cross-entropy loss.

In a very deep plain network the gradient is multiplied by many small numbers on its way back, so it shrinks and the early layers barely learn. A skip connection adds the input to a layer's output, giving the gradient a direct path back to earlier layers.

## Model-choice memo

To: Head of Customer-Support Operations

Recommendation: use the logistic-regression escalation model, flagging a ticket for supervisor review when its probability is at least 0.15.

Why: on 240 held-out tickets, the logistic model at tau = 0.3 reached recall 0.7451 and f1 0.5714, while KNN reached recall 0.3333 and f1 0.3696, and the two voting ensembles (f1 0.5344 and 0.5507) did not beat it. The threshold of 0.15 was chosen on training tickets using your costs (INR 50 per unnecessary review, INR 400 per missed escalation). On the test tickets it costs INR 6050, compared with INR 11800 at the default 0.5 and INR 20400 if nobody is flagged. I did not compute the cost of the other models with this rule.

Risk: the test set holds only 51 escalations and a single split, so the figures could move with a different split. The data is also synthetic and the INR 50 and INR 400 costs are rough estimates, so confirm them before relying on INR 6050.

Next build: neither Model A (order photos) nor Model B (ticket text) should be built before the structured model is in production; Model B is the better second step because ticket text is the main use case.

## Limitations

- The data is synthetic and generated from known formulas, so real Swiggy tickets could behave differently, and results here may be cleaner than reality.
- Everything depends on one fixed 560/240 split; the test set has only 51 escalations, so recall, f1 and cost figures are noisy and the chosen K and tau are valid for this split only.
- The costs (INR 50 and INR 400) are assumed round figures, and the best threshold would move if the real costs differ.
- K-means uses pinned starting rows, so the segments are one reproducible solution, not necessarily the best one, and the hierarchical clustering table is tiny (six cohorts), so its cuts are only illustrative.
- The logistic model, regression and PCA assume linear relationships; tickets with non-linear patterns would be missed.
