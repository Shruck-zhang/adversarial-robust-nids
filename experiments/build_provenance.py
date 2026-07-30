# -*- coding: utf-8 -*-
"""Build a fully granular experiment-provenance table (supervisor point 1): one row per
model x seed x artefact x attack-configuration, so every reported result is traceable to
its exact split, model version, preprocessing, seed, decision rule and output file."""
import os, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import pandas as pd

TAB = os.path.join(os.path.dirname(__file__), "..", "results", "tables")
SPLIT = "leakage-free StratifiedGroupKFold on 2-sig-fig flow fingerprint; 60/20/20 (train 1.10M / val 367k / test 367k)"
PREP = "RobustScaler fit on train + clip(+/-10); 83->66 features (drop zero-var + one of each |corr|>0.98, on train)"
rows = []
def add(**k): rows.append(k)

# 1) full-data baselines / deployed reference
for m, art in [("RandomForest", "sklearn RF (300 trees)"), ("XGBoost", "XGBoost (400 trees, per-sample wts)"),
               ("MLP (sklearn)", "sklearn MLPClassifier 128-64"), ("1D-CNN", "torch 1D-CNN")]:
    add(experiment="baselines", model=m, artefact=art, variant="clean", train_set="FULL 1.10M",
        eval_set="FULL test 367k", seed=42, attack="none", decision_rule="argmax",
        metric="macro-F1/balAcc/MCC/FPR", output_file="models_summary.csv")

# 2) comprehensive comparison (per-class-capped train, single seed)
for m in pd.read_csv(os.path.join(TAB, "model_comparison.csv"))["model"]:
    art = "sklearn MLPClassifier" if m == "MLP" else ("torch net" if m in
          ("DNN", "1D-CNN", "LSTM", "GRU", "CNN-LSTM") else ("pytorch-tabnet" if m == "TabNet" else "sklearn"))
    add(experiment="model_comparison", model=m, artefact=art, variant="clean",
        train_set="capped 25k/class (~200k)", eval_set="FULL test 367k", seed=42, attack="none",
        decision_rule="argmax", metric="macro-F1 etc", output_file="model_comparison.csv")

# 3) repeated-run CIs (per model x seed)
for _, r in pd.read_csv(os.path.join(TAB, "multiseed_ci_rows.csv")).iterrows():
    art = "sklearn MLPClassifier" if r["model"] == "MLP (sklearn)" else (
          "torch net" if r["model"] in ("DNN", "1D-CNN") else "sklearn")
    add(experiment="multiseed_ci", model=r["model"], artefact=art, variant="clean",
        train_set=f"capped 25k/class (seed {int(r['seed'])})", eval_set="FULL test 367k",
        seed=int(r["seed"]), attack="none", decision_rule="argmax", metric=f"macro-F1={r['macro_f1']}",
        output_file="multiseed_ci_rows.csv / multiseed_ci.csv / multiseed_stats.csv")

# 4) robustness comparison (per model x variant); clean on full test, evasion on capped sample
for _, r in pd.read_csv(os.path.join(TAB, "robustness_comparison.csv")).iterrows():
    add(experiment="robustness_comparison", model=r["model"], artefact="torch net",
        variant=r["variant"], train_set="capped 25k/class", eval_set="clean: FULL test; evasion: capped 2k/class+15k benign",
        seed=42, attack="FGSM & PGD, L-inf, eps in {0..0.3}, PGD 10 steps a=eps/4, untargeted, 0 restarts (white-box on the net)",
        decision_rule="argmax", metric="clean macro-F1 + FGSM/PGD evasion@0.1", output_file="robustness_comparison.csv")

# 5) deployed two-tier (full data)
for m, art in [("XGBoost", "full-data XGBoost.pkl"), ("hardened-DNN", "full-data dnn_hardened.pt"),
               ("two-tier", "XGBoost + hardened-DNN, OR-fusion")]:
    add(experiment="two_tier_clean", model=m, artefact=art, variant="clean/hardened", train_set="FULL 1.10M",
        eval_set="FULL test 367k", seed=42, attack="none", decision_rule="argmax (+OR fusion for two-tier)",
        metric="macro-F1/balAcc/FPR", output_file="two_tier_clean.csv")

add(experiment="two_tier_robustness", model="two-tier (XGBoost+hardened-DNN)", artefact="deployed two-tier",
    variant="under attack", train_set="FULL 1.10M", eval_set="capped 2k/class attack + 15k benign", seed=0,
    attack="PGD crafted on the hardened DNN (L-inf, 10 steps, a=eps/4, untargeted); eps in {0,0.02,0.05,0.1,0.15,0.2,0.3}; XGBoost hit by transfer",
    decision_rule="argmax + OR fusion", metric="evasion(DNN / XGB-transfer / two-tier)", output_file="two_tier_robustness.csv")

add(experiment="stronger_dnn_attack", model="two-tier (XGBoost+hardened-DNN)", artefact="deployed two-tier",
    variant="under attack", train_set="FULL 1.10M", eval_set="300/class attack flows (seed 0)", seed=0,
    attack="stronger DNN-targeted PGD: L-inf, 50 steps, a=eps/10, 5 random restarts, untargeted; joint-evasion criterion (both tiers benign); eps in {0.05,0.1,0.2}",
    decision_rule="argmax + OR fusion", metric="two-tier evasion vs transfer", output_file="adaptive_ensemble_attack.csv")

# 6) detection engine + rules (deployed models, raw-feature input)
for sysname, of in [("rule layer (PortScan/DDoS/BruteForce signatures)", "detection_rules.csv"),
                    ("full engine (rules + two-tier AI)", "detection_engine.csv")]:
    add(experiment="detection_layer", model=sysname, artefact="rules + deployed two-tier", variant="clean",
        train_set="FULL (deployed models)", eval_set="FULL test raw 367k", seed=42, attack="none",
        decision_rule="rule thresholds on raw features + argmax + OR fusion", metric="precision/recall/macro-F1",
        output_file=of)

add(experiment="rare_class_attempted", model="full engine", artefact="rules + deployed two-tier",
    variant="clean", train_set="FULL", eval_set="FULL test raw, split by is_attempted flag (metadata, not a feature)",
    seed=42, attack="none", decision_rule="argmax + OR fusion", metric="detection rate: successful vs attempted (+counts)",
    output_file="rare_class_attempted.csv")

add(experiment="deployment_latency", model="two-tier inference / full engine / engine+adapter",
    artefact="deployed", variant="timing", train_set="n/a", eval_set="test flows (warm-up 512 then 500 single-flow / batched)",
    seed="n/a", attack="none", decision_rule="n/a",
    metric="single-flow p50/p90/p99 + batched throughput; Intel i7 16c 17GB CPU-only",
    output_file="deployment_latency.csv / deployment_throughput.csv / deployment_hardware.txt")

df = pd.DataFrame(rows, columns=["experiment", "model", "artefact", "variant", "train_set", "eval_set",
                                 "seed", "attack", "decision_rule", "metric", "output_file"])
df.insert(0, "dataset", "corrected CICIDS2017 (Engelen et al., WTMC 2021), 8-class")
df.insert(1, "split", SPLIT); df.insert(2, "preprocessing", PREP)
df.to_csv(os.path.join(TAB, "experiment_provenance_detailed.csv"), index=False)
print(f"rows: {len(df)}  -> results/tables/experiment_provenance_detailed.csv")
print(df[["experiment", "model", "variant", "seed", "output_file"]].to_string(index=False))
