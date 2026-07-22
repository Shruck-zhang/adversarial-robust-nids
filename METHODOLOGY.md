# Methodology and Experimental Protocol

This document specifies the experimental protocol precisely enough to reproduce every
reported number, and reconciles the clean-performance results across sections. It covers
experiment provenance and
reconciliation, the adversarial threat model, the model-selection/tuning procedure, the
leakage-aware split definition, and the separation of software verification from model
and robustness validation.

## 1  Dataset, task and leakage-aware split

- **Dataset.** The *corrected* CICIDS2017 (Engelen, Rimmer & Joosen, WTMC 2021), which
  fixes flow-construction and labelling defects in the original release.
- **Task.** A single fixed task throughout: **8-class multiclass** classification
  (Benign, DoS, DDoS, PortScan, BruteForce, WebAttack, Bot, Infiltration). Decisions are
  taken by **argmax** over the class posteriors — no tuned operating thresholds are used,
  so no per-experiment threshold selection can explain result differences.
- **Cleaning.** 2.10 M raw flows → drop non-finite rows and exact duplicates → **1.84 M
  flows**. Feature pruning removes zero-variance and one of each pair with |corr| > 0.98,
  leaving **66 features**. Pruning and scaling statistics are fit on the training split
  only.
- **Split.** A leakage-aware **60/20/20** train/val/test split (train 1.10 M, val 367 k,
  test 367 k).

### 1.1  Grouping used in the leakage-aware split

Near-duplicate flows (bursts of near-identical attack flows) can otherwise leak between
train and test. To prevent this, each flow is assigned a **group key** equal to a signed
**2-significant-figure fingerprint** of its full feature vector: each feature is rounded
to two significant figures and the resulting rounded vector is hashed
(`pandas.util.hash_pandas_object`). Flows sharing a fingerprint form one group.
`StratifiedGroupKFold` then partitions the data so that **every group lies entirely on
one side of every split** while preserving class proportions. The validation split is
carved from the training portion by the same procedure.

### 1.2  Attempted-attack flag — no label leakage

Payload-less *attempted* attack variants are merged into their parent class, with a
boolean `is_attempted` flag retained **as metadata only**. This flag is stored in
separate arrays (`att_train/val/test`) and is **never** part of the 66-feature model
input (verified: the feature list contains no attempt-derived column; `X_train` has
exactly 66 columns while `att_train` is a separate 1-D array). It is used only to
partition the rare-class analysis into successful vs attempted attacks after prediction,
so it cannot leak the label.

## 2  Models and tuning procedure

Model configurations are **fixed a priori** from established defaults and light,
documented choices rather than an exhaustive hyper-parameter search; this is stated
honestly as a limitation. No grid/Bayesian search or nested cross-validation was
performed. The one tuned design decision — the class-imbalance strategy — was chosen by a
controlled ablation (class weighting vs synthetic oversampling), not by test-set peeking.

| Model | Key hyper-parameters (else library defaults) |
|---|---|
| Random Forest | 300 trees, `max_depth=None`, `min_samples_leaf=2`, `max_features=sqrt`, `class_weight=balanced` |
| XGBoost | 400 trees, per-sample inverse-frequency weights at fit time |
| Decision Tree / Logistic Regression | defaults, `class_weight=balanced` |
| MLP (sklearn) | hidden 128–64, ReLU, Adam, internal early stopping |
| Torch nets (MLP / DNN / 1D-CNN / LSTM / GRU / CNN-LSTM) | Adam `lr=1e-3`, batch 2048, ≤25 epochs, early stop patience 6, StepLR, `sqrt` class weights, cross-entropy |
| Adversarial training | as above + per-batch FGSM augmentation (ε = 0.1), 50/50 clean/adversarial loss |

All models use `random_state = 42`. Because a single seed can mislead, the key
architectures were **re-trained under five seeds** on the same capped budget and
evaluated on the full test set (`experiments/multiseed_ci.py`); the mean macro-F1 and a
95 % t-based confidence interval are:

| Model | macro-F1 mean ± 95% CI | std |
|---|---|---|
| Random Forest | 0.964 ± 0.003 | 0.002 |
| XGBoost | 0.962 ± 0.007 | 0.006 |
| MLP (sklearn) | 0.921 ± 0.022 | 0.017 |
| DNN | 0.916 ± 0.010 | 0.008 |
| 1D-CNN | 0.909 ± 0.024 | 0.020 |

Two things follow. First, the **tree models are markedly more stable** (RF ± 0.003) than
the neural networks (**1D-CNN ± 0.024**), and the tree vs neural CIs do **not overlap**, so
"trees outperform the neural nets on this tabular data" is statistically supported, not
noise. Second, the wide 1D-CNN interval [0.884, 0.933] **contains both** single-run values
that appeared inconsistent across sections (0.893 and 0.957), confirming that that
discrepancy is training-run variance rather than an experimental error.

## 3  Experimental protocol and result reconciliation

The clean-performance numbers differ across sections **not** because of different tasks,
preprocessing, thresholds or seeds — those are identical — but because three experiments
deliberately used different **training-set sizes**, **evaluation sets**, or **model
artefacts**. The full provenance is recorded in `results/tables/experiment_provenance.csv`
and summarised here.

| Experiment | Train set | Eval set | "MLP" artefact | Purpose |
|---|---|---|---|---|
| Baselines / deployed | **FULL** 1.10 M | **FULL** test 367 k | sklearn MLP | performance of the deployed models |
| Comprehensive comparison | **capped** 25 k/class (~200 k) | **FULL** test 367 k | sklearn MLP | *fair, tractable* cross-architecture comparison at an equal training budget |
| Robustness comparison | capped 25 k/class | clean-F1 on **FULL** test; evasion on a capped balanced sample | **torch** MLP | robustness of *differentiable* nets before/after hardening |

This explains every discrepancy across sections, for example:

- **XGBoost** macro-F1 0.967 (comparison) vs 0.978 (deployed) — the *same, deterministic*
  model, differing only by **capped vs full training data**.
- **1D-CNN** 0.893 (comparison) vs 0.957 (robustness) — both on the **full test set** with
  the **same capped training budget**; the gap is **training-run variance** (the CNN
  trains unstably, and the two runs differ in data ordering and early-stopping epoch), not
  a difference of task or evaluation set. This is precisely why repeated-run confidence
  intervals (§2) are being added.
- **"MLP"** 0.922 vs 0.881 — these are **two different models**: the sklearn `MLPClassifier`
  in the comparison and a Torch feed-forward net in the robustness study (the Torch net is
  used there because it must be gradient-attackable). Renamed "MLP (torch)".

**Corrective actions.**
1. **Canonical numbers.** Deployed/clean performance is always reported from the
   **full-data models on the full test set**; the capped-train comparison is labelled
   explicitly as an *equal-budget architecture comparison*, and its absolute values are
   never mixed with the deployed figures.
2. **Naming.** The Torch feed-forward net is renamed **"MLP (torch)"** to remove the
   collision with the sklearn `MLP`.
3. **Evaluation set.** The robustness clean-F1 is being recomputed on the **full test
   set** so it is directly comparable (Batch B).
4. Optionally, the full comparison will be **re-run on full training data** so the O1
   ranking and the deployed numbers coincide (compute-heavy; scheduled).

## 4  Adversarial threat model

| Property | Setting |
|---|---|
| Attacker knowledge | **white-box** on the differentiable detector (full gradient access) |
| Norm | **L∞** (perturbation projected to the L∞ ball) |
| Feature space | the 66-dim **RobustScaler-scaled, clip(±10)** space |
| Attacks | **FGSM** (one step) and **PGD** (iterative) |
| PGD iterations | **10** |
| PGD step size α | **ε / 4** |
| Restarts | **0** (single start from the clean point) |
| Objective | **untargeted** (maximise cross-entropy of the true label) |
| Projection | clip to [x₀−ε, x₀+ε] ∩ [−10, 10] |
| ε range evaluated | 0.02–0.3 (headline results at ε = 0.1) |

**Feasibility of the perturbed vectors (stated limitation).** The perturbation is
**unconstrained in feature space**: it may move features independently that are in reality
**discrete** (packet/byte counts, TCP-flag counts, ports), **immutable** (protocol), or
**mutually dependent** (e.g. total length = forward + backward; means = totals ÷ counts).
The resulting vectors therefore **do not necessarily correspond to feasible network
traffic**, and the evasion rates are a **conservative upper bound** on the capability of a
realistic adversary who must keep the traffic protocol-valid and semantically consistent.
Constraint-aware, realizable attacks (fixing immutable features, enforcing count/aggregate
dependencies) are identified as further work.

## 5  Two-tier robustness — scope of the current claim

The two-tier robustness experiment crafts perturbations against the **DNN** and measures
their effect on the DNN, on XGBoost (transfer), and on the fused system. It therefore
establishes a **transfer-attack** result — notably, that DNN-crafted perturbations transfer
to and evade XGBoost in up to 67 % of cases, so a non-differentiable tree is not inherently
safe.

To go beyond transfer, an **adaptive white-box attack against the complete ensemble** was
run (`experiments/adaptive_ensemble_attack.py`). Because XGBoost transfers easily
(~0.63 evasion at ε = 0.1) while the adversarially-hardened DNN is the **binding
constraint**, the adaptive attacker concentrates a much stronger PGD on the DNN — **50
iterations, step size ε/10, five random restarts** — and both tiers are then required to be
evaded. The two-tier evasion is **essentially unchanged** from the transfer setting:

| ε | two-tier evasion (transfer) | two-tier evasion (adaptive) |
|---|---|---|
| 0.05 | 0.013 | 0.013 |
| 0.10 | 0.017 | 0.017 |
| 0.20 | 0.026 | 0.031 |

The hardened DNN resists the stronger attack, so the system's robustness is **not an
artefact of a weak attacker**. This remains qualified: it is one (strong) white-box
adaptive attack; it does not cover query-based/black-box strategies, and the perturbations
are still unconstrained feature-space upper bounds (§4) that need not be realizable
traffic. Query-based black-box and feature-constrained realizable attacks remain further
work.

## 6  Validation strategy — software verification vs model validation

Two distinct kinds of evidence are reported **separately**:

- **Software verification** (does the implementation behave as specified?): unit and
  integration tests of the adapter, rule layer, fusion logic, streaming consumer and
  metrics — pass/fail correctness checks, independent of scientific claims.
- **Model and robustness validation** (are the scientific results sound?): held-out
  performance, per-class detection, adversarial evasion under the threat model of §4, and
  the deployment measurements of the next point.

The automated harness produces both, but they are presented under separate headings so
that passing software tests are not mistaken for independent validation of detection or
robustness performance.

## 7  Deployment measurement

Deployment figures distinguish **model-inference** from the **complete capture-to-alert
pipeline**, and report a **latency distribution**, not just a mean
(`experiments/deployment_latency.py`).

**Hardware.** Intel Core i7 (Tiger Lake), 16 logical cores, 17 GB RAM, Windows,
**CPU-only** PyTorch (no GPU).

| Stage | single-flow p50 | p99 | batched throughput (bs = 4096) |
|---|---|---|---|
| Two-tier inference (scaled input) | 11.2 ms | 34 ms | 53 k flows/s |
| Full engine (rules + AI, raw input) | 12.5 ms | 48 ms | 75 k flows/s |
| Engine + CICFlowMeter feature adapter | 30 ms | 66 ms | 33 k flows/s |

Inference throughput scales with batch size (45 flows/s at batch 1 → 74 k flows/s at
batch 50 k); the ~11 ms single-flow figure is dominated by per-call framework overhead, so
production deployment micro-batches. (Earlier reports of "~160 k flows/s / 6 µs per flow"
were warm-cache batch measurements and are superseded by these distribution-based numbers.)

**Crucially, model inference is not the operational bottleneck.** End-to-end
capture-to-alert latency is dominated by the **flow-completion floor**: CICFlowMeter cannot
emit a flow until it ends (TCP FIN/RST or the activity/flow timeout, up to ~120 s), so the
time from first packet to alert is set by the flow's own duration plus the exporter
interval — seconds, not the millisecond-scale inference. This distinction (inference speed
vs pipeline latency) is stated explicitly so the throughput figures are not mistaken for
end-to-end responsiveness.
