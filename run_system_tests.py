# -*- coding: utf-8 -*-
"""Fully-automated system test harness for the two-layer IDS.

One command runs the whole test pyramid and emits a graded report:

    L1 unit + L2 integration   -> pytest (tests/)            [correctness + regression]
    L3 performance / SLA       -> throughput + latency        [real-time capability]
    L4 robustness / adversarial-> PGD evasion + malformed I/O  [security + resilience]
    L5 attack simulation       -> per-class replay detection   [does it catch attacks]

Outputs results/test_reports/report_<timestamp>.{json,csv,html} and returns a
non-zero exit code if any acceptance criterion fails.
"""
import os, sys, time, json, csv, platform, subprocess, datetime
import xml.etree.ElementTree as ET
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, os.path.join(ROOT, "tests"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import joblib
from detection_engine import DetectionEngine
import realtime as RT
import security as S
from thresholds import THRESHOLDS as T, EPS_PGD, SEED
import pandas as pd

PROC = os.path.join(ROOT, "data", "processed")
MODELS = os.path.join(ROOT, "results", "models")
SAMPLE = os.path.join(ROOT, "data", "sample_flows", "sample_cicflowmeter_Flow.csv")
REPORTS = os.path.join(ROOT, "results", "test_reports")
os.makedirs(REPORTS, exist_ok=True)

results = []

# Software verification is reported separately from
# independent model / robustness validation. Each layer maps to one category.
CATEGORY = {
    "L1-unit": "A. Software verification", "L2-integration": "A. Software verification",
    "L3-perf": "B. Model & robustness validation", "L4-robust": "B. Model & robustness validation",
    "L5-attack": "B. Model & robustness validation",
}
def category(layer):
    return CATEGORY.get(layer, "B. Model & robustness validation")

def add(layer, name, value, threshold, op, unit="", detail=""):
    ok = {">=": lambda: value >= threshold, "<=": lambda: value <= threshold,
          "bool": lambda: bool(value)}[op]()
    results.append({"layer": layer, "name": name, "value": value, "threshold": threshold,
                    "op": op, "unit": unit, "passed": bool(ok), "detail": detail})
    tag = "PASS" if ok else "FAIL"
    vs = f"{value:.4f}" if isinstance(value, float) else str(value)
    print(f"  [{tag}] {layer:16s} {name:34s} {vs} {unit} (thr {op} {threshold})")
    return ok


# ============================ L1 + L2 via pytest ============================ #
def run_pytest():
    print("\n== L1 unit + L2 integration (pytest) ==")
    junit = os.path.join(REPORTS, "_junit.xml")
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "-p", "no:cacheprovider",
                    f"--junitxml={junit}"], cwd=ROOT,
                   env=dict(os.environ, PYTHONIOENCODING="utf-8"),
                   capture_output=True, text=True, timeout=900)
    tree = ET.parse(junit); root = tree.getroot()
    suite = root.find("testsuite") if root.tag == "testsuites" else root
    for tc in suite.findall("testcase"):
        fname = (tc.get("classname") or "").split(".")[-1]
        layer = "L1-unit" if "unit" in fname else "L2-integration"
        failed = tc.find("failure") is not None or tc.find("error") is not None
        msg = ""
        if failed:
            node = tc.find("failure") if tc.find("failure") is not None else tc.find("error")
            msg = (node.get("message") or "")[:160]
        results.append({"layer": layer, "name": tc.get("name"), "value": "fail" if failed else "pass",
                        "threshold": "pass", "op": "==", "unit": "", "passed": not failed, "detail": msg})
    npass = sum(1 for r in results if r["layer"].startswith("L1") or r["layer"].startswith("L2") and r["passed"])
    total = sum(1 for r in results if r["layer"].startswith("L1") or r["layer"].startswith("L2"))
    ok = sum(1 for r in results if (r["layer"].startswith("L1") or r["layer"].startswith("L2")) and r["passed"])
    print(f"  pytest: {ok}/{total} passed")


# ============================ L3 performance ============================ #
def run_perf(engine, Xraw):
    print("\n== L3 performance / real-time SLA ==")
    engine.predict(Xraw[:256])                                   # warm up
    n = min(50000, len(Xraw)); batch = Xraw[:n]
    t = time.time(); engine.predict_detailed(batch); dt = time.time() - t
    add("L3-perf", "batch throughput", n / dt, T["batch_throughput_min"], ">=", "flows/s")
    m = 300
    t = time.time()
    for i in range(m):
        engine.predict_detailed(Xraw[i:i + 1])
    lat = 1000 * (time.time() - t) / m
    add("L3-perf", "single-flow latency", lat, T["single_latency_ms_max"], "<=", "ms/flow")
    try:
        import psutil
        rss = psutil.Process().memory_info().rss / 1e6
        results.append({"layer": "L3-perf", "name": "process memory (info)", "value": round(rss, 1),
                        "threshold": "-", "op": "bool", "unit": "MB", "passed": True, "detail": ""})
        print(f"  [INFO] L3-perf         process memory                    {rss:.1f} MB")
    except Exception:
        pass


# ============================ L4 robustness ============================ #
def run_robustness(engine):
    print("\n== L4 robustness / adversarial ==")
    Xs = np.load(os.path.join(PROC, "X_test.npy")).astype("float32")
    y = np.load(os.path.join(PROC, "y_test.npy"))
    ben = engine.benign
    rng = np.random.RandomState(SEED)
    parts = [rng.choice(np.where(y == c)[0], min(2000, int((y == c).sum())), replace=False)
             for c in range(len(engine.class_names)) if c != ben]
    sel = np.concatenate(parts); Xa, ya = Xs[sel], y[sel]
    Xadv = S.pgd_grad(engine.ai.dnn, Xa, ya, EPS_PGD)
    dnn_p = engine.ai.dnn.predict(Xadv); xgb_p = engine.ai.xgb.predict(Xadv)
    ev_dnn = float((dnn_p == ben).mean())
    ev_two = float(((dnn_p == ben) & (xgb_p == ben)).mean())
    add("L4-robust", f"hardened-DNN PGD evasion @e={EPS_PGD}", ev_dnn, T["dnn_pgd_evasion_max"], "<=")
    add("L4-robust", f"two-tier PGD evasion @e={EPS_PGD}", ev_two, T["two_tier_pgd_evasion_max"], "<=")

    # malformed / adversarial INPUT must be handled without crashing
    fn = engine.feature_names
    cases = {
        "empty dataframe": pd.DataFrame(),
        "wrong columns": pd.DataFrame({"foo": [1, 2], "bar": [3, 4], "Flow Duration": [5, 6]}),
        "all NaN/Inf": pd.DataFrame(np.full((3, len(fn)), np.inf), columns=fn).assign(**{"Flow Duration": 10}),
        "extreme + negative": pd.DataFrame(np.full((3, len(fn)), -1e18), columns=fn).assign(**{"Flow Duration": 10}),
        "string junk": pd.DataFrame([["x"] * len(fn)], columns=fn).assign(**{"Flow Duration": 10}),
    }
    ids = RT.RealTimeIDS(engine); handled = True; failed_case = ""
    for label, df in cases.items():
        try:
            ids.score_dataframe(df)
        except Exception as e:
            handled = False; failed_case = f"{label}: {type(e).__name__}"
            break
    add("L4-robust", "malformed input handled", handled, True, "bool", detail=failed_case)


# ============================ L5 attack simulation ============================ #
def run_attack_sim(engine, Xraw, y):
    print("\n== L5 attack simulation (per-class replay) ==")
    d = engine.predict_detailed(Xraw)
    ben = engine.benign
    for name, floor in T["per_class_detect_min"].items():
        c = engine.class_names.index(name)
        m = y == c
        if not m.any():
            continue
        rate = float(d["is_attack"][m].mean())
        add("L5-attack", f"detect {name}", rate, floor, ">=", detail=f"n={int(m.sum())}")


# ============================ report ============================ #
def write_reports(meta):
    ts = meta["timestamp_file"]
    base = os.path.join(REPORTS, f"report_{ts}")
    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2, default=str)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["layer", "name", "value", "threshold", "op", "unit", "passed", "detail"])
        w.writeheader(); w.writerows(results)
    _write_html(base + ".html", meta)
    return base


def _write_html(path, meta):
    n_pass = sum(r["passed"] for r in results); n = len(results)
    grade = "PASS" if n_pass == n else "FAIL"
    colour = "#2E7D32" if grade == "PASS" else "#C62828"
    rows = []
    for cat in ["A. Software verification", "B. Model & robustness validation"]:
        crs = [r for r in results if category(r["layer"]) == cat]
        if not crs:
            continue
        cp = sum(r["passed"] for r in crs)
        rows.append(f"<tr style='background:#37474F;color:#fff'><td colspan='6'>"
                    f"<b>{cat}</b> &nbsp; ({cp}/{len(crs)} passed)</td></tr>")
        for r in sorted(crs, key=lambda x: (x["layer"], x["name"])):
            pc = "#E8F5E9" if r["passed"] else "#FFEBEE"
            tc = "#2E7D32" if r["passed"] else "#C62828"
            val = f"{r['value']:.4f}" if isinstance(r["value"], float) else r["value"]
            rows.append(f"<tr style='background:{pc}'><td>{r['layer']}</td><td>{r['name']}</td>"
                        f"<td style='text-align:right'>{val} {r['unit']}</td>"
                        f"<td style='text-align:right'>{r['op']} {r['threshold']}</td>"
                        f"<td style='color:{tc};font-weight:600'>{'PASS' if r['passed'] else 'FAIL'}</td>"
                        f"<td style='color:#666;font-size:12px'>{r['detail']}</td></tr>")
    html = f"""<!doctype html><meta charset="utf-8"><title>IDS system test report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;margin:32px;color:#222}}
h1{{margin:0 0 4px}} .sub{{color:#666;margin-bottom:18px}}
.badge{{display:inline-block;padding:6px 18px;border-radius:8px;color:#fff;background:{colour};font-weight:700;font-size:20px}}
table{{border-collapse:collapse;width:100%;margin-top:16px;font-size:14px}}
th,td{{padding:7px 10px;border-bottom:1px solid #eee;text-align:left}}
th{{background:#37474F;color:#fff}} .meta{{font-size:13px;color:#555;margin-top:8px}}</style>
<h1>Two-layer IDS — automated system test report</h1>
<div class="sub">{meta['timestamp']}</div>
<div class="badge">{grade}</div> &nbsp; <b>{n_pass}/{n}</b> checks passed
<div class="meta">host {meta['host']} · python {meta['python']} · {meta['platform']}</div>
<table><tr><th>Layer</th><th>Check</th><th>Value</th><th>Threshold</th><th>Result</th><th>Detail</th></tr>
{''.join(rows)}</table>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    t0 = time.time()
    now = datetime.datetime.now()
    print("=" * 74)
    print("  Two-layer IDS — automated system test harness")
    print("=" * 74)
    engine = DetectionEngine(MODELS, PROC)
    Xraw = np.load(os.path.join(PROC, "X_test_raw.npy"))
    y = np.load(os.path.join(PROC, "y_test.npy"))

    run_pytest()
    run_perf(engine, Xraw)
    run_robustness(engine)
    run_attack_sim(engine, Xraw, y)

    meta = {"timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp_file": now.strftime("%Y%m%d_%H%M%S"),
            "host": platform.node(), "python": platform.python_version(),
            "platform": platform.platform(), "duration_s": round(time.time() - t0, 1)}
    base = write_reports(meta)

    n_pass = sum(r["passed"] for r in results); n = len(results)
    grade = "PASS" if n_pass == n else "FAIL"
    print("\n" + "=" * 74)
    for cat in ["A. Software verification", "B. Model & robustness validation"]:
        crs = [r for r in results if category(r["layer"]) == cat]
        if crs:
            print(f"  {cat}: {sum(r['passed'] for r in crs)}/{len(crs)} passed")
    print(f"  OVERALL: {grade}   {n_pass}/{n} checks passed   ({meta['duration_s']}s)")
    print(f"  report -> {base}.html / .json / .csv")
    print("=" * 74)
    sys.exit(0 if grade == "PASS" else 1)


if __name__ == "__main__":
    main()
