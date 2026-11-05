"""Assertion tests for ecg_use_case_utility.py (labels, criteria logic, utility statistics).
Run: python test_module.py   (expects ecg_use_case_utility.py in the same folder)"""
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))   # module next to this file
import ecg_use_case_utility as m  # noqa: E402

# ---- labels ---------------------------------------------------------------------------
sup_counts = pd.Series([v[0] for v in m.DIAGNOSTIC.values()]).value_counts().to_dict()
assert sup_counts == {"MI": 14, "STTC": 13, "CD": 11, "HYP": 5, "NORM": 1}, sup_counts
assert len(set(m.DIAGNOSTIC) & set(m.FORM)) == 4          # NDT, NST_, DIG, LNGQT are diagnostic and form
assert set(m.DIAGNOSTIC) & set(m.FORM) == {"NDT", "NST_", "DIG", "LNGQT"}
assert not set(m.RHYTHM) & (set(m.DIAGNOSTIC) | set(m.FORM))
sub = sorted({s for _, s in m.DIAGNOSTIC.values()})
assert len(sub) == 23, len(sub)                            # PTB-XL has 23 diagnostic subclasses

scp = pd.Series(["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 80.0, 'SR': 0.0}", "{'CLBBB': 100.0, 'AFIB': 0.0}",
                 "{'NORM': 50.0, 'LAFB': 100.0, 'SR': 0.0}", "{'SR': 0.0}"])
assert m.build_triage_labels(scp).tolist() == [0, 1, 1, 1, -1]
assert m.build_triage_labels(scp, conflict="exclude").tolist() == [0, 1, 1, -1, -1]
scp_af = pd.Series(["{'NORM': 100.0, 'AFIB': 0.0}", "{'AFIB': 100.0}"])
assert m.build_triage_labels(scp_af).tolist() == [0, -1]                      # superclass level only
assert m.build_triage_labels(scp_af, abnormal_rhythms=("AFIB",)).tolist() == [1, 1]
assert m.build_labels(scp, m.TASKS["triage_ribeiro6"]).tolist()[2] == [0, 0, 1, 0, 1, 0]
assert m.build_labels(scp, m.TASKS["acs_mi_location"]).tolist()[1] == [0, 1, 0, 0]
# likelihood filter: IMI at 80 is dropped with min_likelihood=100
assert m.build_labels(scp, m.TASKS["acs_superclass"], min_likelihood=100).tolist()[1] == [0, 0]
# SR has likelihood 0 (unknown) but is kept by default
assert m.build_labels(scp, m.TASKS["rhythm"]).tolist()[0][m.RHYTHM.index("SR")] == 1

# ---- criteria logic ----------------------------------------------------------------------
c = m.Criteria()
assert m._contiguous({"II": True, "aVF": True}, c.groups)
assert not m._contiguous({"II": True, "V2": True}, c.groups)
lead = lambda st=0.0, t=0.1, q=0.0, qd=0.0, r=1.0, rpt=40.0: {
    "st_j_mV": st, "t_amp_mV": t, "q_amp_mV": q, "q_dur_ms": qd, "r_amp_mV": r,
    "q_r_ratio": abs(q) / r if r > 0 else float("inf"), "r_peak_time_ms": rpt}
base = {l: lead() for l in m.LEADS_12}
f = {"hr_bpm": 70, "pr_ms": 210, "qrs_ms": 100, "axis_deg": -60, "leads": dict(base)}
out = m.apply_criteria(f)
assert out["first_degree_avb"] and out["lafb_axis"] and not out["lpfb_axis"]
assert not out["stemi_st_criteria"] and not out["st_depression"] and not out["pathological_q"]
# inferior STE 0.12 mV in II and aVF -> positive; V2/V3 thresholds need sex and age
f2 = {**f, "leads": {**base, "II": lead(st=0.12), "aVF": lead(st=0.12)}}
assert m.apply_criteria(f2)["stemi_st_criteria"]
f3 = {**f, "leads": {**base, "V2": lead(st=0.18), "V3": lead(st=0.18)}}
assert not m.apply_criteria(f3)["stemi_st_criteria"]                       # sex unknown -> V2/V3 skipped
assert m.apply_criteria(f3, female=True)["stemi_st_criteria"]             # women: >= 0.15 mV
assert not m.apply_criteria(f3, female=False, age=35)["stemi_st_criteria"]  # men < 40: >= 0.25 mV
assert not m.apply_criteria(f3, female=False, age=55)["stemi_st_criteria"]  # men >= 40: >= 0.20 mV
# pathological Q: Q/R >= 0.25 in two contiguous leads; III and aVR are ignored (Sharma et al., 2017)
f4 = {**f, "leads": {**base, "II": lead(q=-0.3), "aVF": lead(q=-0.3)}}
assert m.apply_criteria(f4)["pathological_q"]
f5 = {**f, "leads": {**base, "III": lead(q=-0.3), "aVF": lead(q=-0.3)}}
assert not m.apply_criteria(f5)["pathological_q"]
# T inversion: V1 is excluded, so V1 + V2 does not count, V2 + V3 does
f7 = {**f, "leads": {**base, "V1": lead(t=-0.2), "V2": lead(t=-0.2)}}
assert not m.apply_criteria(f7)["t_inversion"]
f8 = {**f, "leads": {**base, "V2": lead(t=-0.2), "V3": lead(t=-0.2)}}
assert m.apply_criteria(f8)["t_inversion"]
# ST depression > 0.05 mV in any two leads
f9 = {**f, "leads": {**base, "I": lead(st=-0.06), "V4": lead(st=-0.06)}}
assert m.apply_criteria(f9)["st_depression"]
# complete RBBB-like: QRS >= 120 and R-peak time in V1 > 50 ms
f6 = {**f, "qrs_ms": 130, "leads": {**base, "V1": lead(rpt=70)}}
o6 = m.apply_criteria(f6)
assert o6["rbbb_partial_criteria"] and not o6["ivcd_nonspecific"] and not o6["lafb_axis"]

# ---- utility statistics ------------------------------------------------------------------
rng = np.random.default_rng(1)
y = rng.integers(0, 2, 600)
a = y + rng.normal(0, 0.9, 600)
b = y + rng.normal(0, 1.2, 600)
d = m.delong_test(y, a, b)
assert abs(d["auc_a"] - roc_auc_score(y, a)) < 1e-12 and abs(d["auc_b"] - roc_auc_score(y, b)) < 1e-12
# ties: midranks must reproduce sklearn's AUC
yt = rng.integers(0, 2, 300)
at = np.round(yt + rng.normal(0, 1, 300))
bt = np.round(yt + rng.normal(0, 1.5, 300))
dt = m.delong_test(yt, at, bt)
assert abs(dt["auc_a"] - roc_auc_score(yt, at)) < 1e-12
# DeLong SE of the AUC difference vs paired bootstrap SE
diff_boot = []
for _ in range(2000):
    i = rng.integers(0, 600, 600)
    if 0 < y[i].sum() < 600:
        diff_boot.append(roc_auc_score(y[i], a[i]) - roc_auc_score(y[i], b[i]))
se_boot = float(np.std(diff_boot))
se_delong = (d["auc_a"] - d["auc_b"]) / d["z"]
assert abs(se_delong - se_boot) / se_boot < 0.15, (se_delong, se_boot)
# paired bootstrap CI contains the point estimate; non-inferiority logic
Y = np.c_[y, 1 - y]
A, B = np.c_[a, -a], np.c_[b, -b]
pt, lo, hi = m.paired_delta_ci(Y, A, B, n_boot=500)
assert lo <= pt <= hi
assert m.non_inferior(0.019, 0.02) and not m.non_inferior(0.021, 0.02)
# feature fidelity: identical samples -> zero distance
ff = m.feature_fidelity([1, 2, 3, 4], [1, 2, 3, 4], tolerance=0.5)
assert ff["wasserstein"] == 0 and ff["ks_stat"] == 0
r1 = np.random.default_rng(3).normal(160, 20, 2000)
assert m.feature_fidelity(r1, r1 + 2, tolerance=10)["within_tolerance"]          # shift 2 ms, CI inside +/-10
assert not m.feature_fidelity(r1, r1 + 9, tolerance=10)["within_tolerance"]      # shift 9 ms, CI crosses 10
# criterion rates: None (not evaluable) is dropped and counted
g = m.criterion_rate_gap([True, False, None, True], [True, True, None, None], margin=0.5)
assert g["not_evaluable_real"] == 1 and g["not_evaluable_synth"] == 2 and abs(g["gap"] - (1 - 2 / 3)) < 1e-12
assert abs(m.measurement_noise([100, 110, 120], [104, 110, 130], q=0.5) - 4) < 1e-12
print("all assertions passed; n = 600 simulated cases: DeLong SE %.4f vs paired-bootstrap SE %.4f" % (se_delong, se_boot))
