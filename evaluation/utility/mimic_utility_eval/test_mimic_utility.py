"""Assertion tests for mimic_utility.py (layout, label rules, statistics, fidelity and privacy metrics).

Run:  python test_mimic_utility.py           unit tests on hand-built tensors (about 1 minute)
      python test_mimic_utility.py --real    also reproduce the review's numbers on our preprocessed files
                                             (file locations in paths.py)
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")    # avoid OpenMP oversubscription when other jobs share the CPU
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mimic_utility as M  # noqa: E402

K = {s: i for i, s in enumerate(M.SIG)}


def blank(n=1):
    return np.zeros((n, 16, 72))


def put(X, i, sig, hours, values):
    X[i, 2 * K[sig], hours] = values
    X[i, 2 * K[sig] + 1, hours] = 1


def labels_of(X, fn):
    v, m = M.split(X)
    return fn(v, m)


# ---- 0. layout ---------------------------------------------------------------------------------------
X = blank(2)
put(X, 0, "heart_rate", [0, 1], [80, 90])
v, m = M.split(X)
assert m["heart_rate"][0, :2].all() and not m["heart_rate"][0, 2:].any() and v["heart_rate"][0, 1] == 90
assert M.CHANNELS[:4] == ["heart_rate", "heart_rate_mask", "sbp", "sbp_mask"] and len(M.CHANNELS) == 16
Xs, notes = M.prepare_synthetic(X.transpose(0, 2, 1))            # stays x time x channels is transposed back
assert np.allclose(Xs, X) and any("transposed" in n for n in notes)
Xc = X.copy(); Xc[:, 1::2] = 0.6 * Xc[:, 1::2] + 0.2              # continuous masks 0.2 / 0.8 -> 0 / 1
assert np.array_equal(M.prepare_synthetic(Xc)[0][:, 1::2], X[:, 1::2])
Xv, notes = M.prepare_synthetic(np.full((3, 8, 72), 100.0))       # values only -> masks all 1, flagged
assert Xv.shape == (3, 16, 72) and (Xv[:, 1::2] == 1).all() and any("values only" in n for n in notes)
Xn = X.copy(); Xn[0, 0, 0] = np.nan
Xn2, notes = M.prepare_synthetic(Xn)
assert Xn2[0, 1, 0] == 0 and np.isfinite(Xn2).all() and any("non-finite" in n for n in notes)
try:
    M.prepare_synthetic(np.zeros((2, 10, 72))); raise AssertionError("bad shape accepted")
except ValueError:
    pass
ch = M.load_channels(None)
assert ch["clip_bounds"]["spo2"] == (50.0, 100.0) and ch["leading_fill"]["sbp"] == 115.5

Xr = blank(3)
put(Xr, 0, "sodium", [0, 5], [134, 135]); put(Xr, 1, "sodium", [3], [140]); put(Xr, 2, "spo2", [1, 2], [89.5, 90])
Xg = blank(1)
put(Xg, 0, "sodium", [0, 1, 2, 3], [134.97, 135.4, 120.0, 170.0]); put(Xg, 0, "spo2", [0, 1], [89.7, 89.76])
snapped, shift = M.ValueGrid(Xr).snap(Xg)
assert snapped[0, 2 * K["sodium"], :4].tolist() == [135, 135, 120.0, 170.0]   # in range -> nearest real value; outside -> unchanged
assert snapped[0, 2 * K["spo2"], :2].tolist() == [89.5, 90]
assert shift["sodium"] > 0

# ---- 1. label rules ----------------------------------------------------------------------------------
assert M.longest_run(np.array([[1, 1, 0, 1, 1, 1, 0]], bool))[0] == 3
X = blank(2)
put(X, 0, "sbp", [0, 1], [85, 85])                                # two consecutive measured hours < 90
put(X, 1, "sbp", [0, 2], [85, 85]); X[1, 2 * K["sbp"], 1] = 80     # hour 1 has a low value but is NOT measured
assert labels_of(X, lambda v, m: M.sustained(v, m, "sbp", "<", 90)).tolist() == [True, False]

X = blank(3)
for h in (3, 4):
    put(X, 0, "sbp", h, 80); put(X, 0, "dbp", h, 55)               # MAP 63.3 for 2 h
    put(X, 1, "sbp", h, 85); put(X, 1, "dbp", h, 55)               # MAP exactly 65 -> not < 65
put(X, 2, "sbp", 3, 80); put(X, 2, "dbp", 3, 55)                   # only 1 hour
assert labels_of(X, M.hypotension).tolist() == [True, False, False]
assert abs(M.derived_map({"sbp": np.array(120.0), "dbp": np.array(60.0)}) - 80.0) < 1e-12

X = blank(3)
for h in (10, 11):
    put(X, 0, "heart_rate", h, 120); put(X, 0, "sbp", h, 110)       # SI 1.09
    put(X, 1, "heart_rate", h, 100); put(X, 1, "sbp", h, 100)       # SI exactly 1.0 counts (>= 1.0)
put(X, 2, "heart_rate", [10, 11], [130, 130]); put(X, 2, "sbp", 10, 100)   # SBP missing at hour 11
assert labels_of(X, M.shock_index_event).tolist() == [True, True, False]


def cr(*pairs):
    X = blank(1)
    for h, c in pairs:
        put(X, 0, "creatinine", h, c)
    v, m = M.split(X)
    return {b: bool(M.aki_creatinine(v, m, b)[0]) for b in ("rolling", "first", "minimum")}, int(M.aki_onset(v, m)[0])


assert cr((0, 1.0), (10, 1.3)) == ({"rolling": True, "first": True, "minimum": True}, 10)          # +0.3 in 48 h
assert cr((0, 1.0), (10, 1.2), (60, 1.35)) == ({"rolling": False, "first": False, "minimum": False}, -1)  # rise too slow
assert cr((0, 2.0), (30, 1.2)) == ({"rolling": False, "first": False, "minimum": True}, -1)        # recovery: only 'minimum' fires
assert cr((0, 1.0), (60, 1.6)) == ({"rolling": True, "first": True, "minimum": True}, 60)          # 1.6 >= 1.5 x 1.0
assert cr((0, 1.0)) == ({"rolling": False, "first": False, "minimum": False}, -1)                 # one value: no AKI
assert cr((0, 1.4), (5, 1.0), (20, 1.35)) == ({"rolling": True, "first": False, "minimum": True}, 20)  # rolling uses lowest earlier
X = blank(3)
for i, (h, c) in enumerate([(10, 1.5), (30, 1.5), (None, None)]):
    put(X, i, "creatinine", 0, 1.0)
    if h is not None:
        put(X, i, "creatinine", h, c)
y = M.TASKS["aki_new_h25_72"].label_fn(X)
assert np.isnan(y[0]) and y[1] == 1 and y[2] == 0                 # AKI in hours 0-24 excluded; new AKI later = 1

X = blank(2)
put(X, 0, "glucose", 5, 60); X[1, 2 * K["glucose"], 5] = 60       # stay 1: low value but not measured
assert labels_of(X, lambda v, m: M.any_measured(v, m, "glucose", "<", 70)).tolist() == [True, False]

na = np.array([124.9, 125, 129.9, 130, 134.9, 135, 145, 145.1, 150, 150.1, 155, 155.1, np.nan])
assert M.funk_grade(na).tolist() == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, -1]
X = blank(2)
put(X, 0, "glucose", [1, 5, 9], [100, 120, 140]); put(X, 1, "glucose", [1, 5], [100, 300])
v, m = M.split(X)
gsd = M.glucose_sd(v, m)
assert abs(gsd[0] - 20.0) < 1e-9 and np.isnan(gsd[1])            # SD with ddof 1; < 3 values -> NaN
put(X, 0, "sodium", [7, 40], [131, 140])
v, m = M.split(X)
assert M.first_value(v, m, "sodium")[0] == 131 and np.isnan(M.first_value(v, m, "sodium")[1])
assert M.last_measured(v["glucose"], m["glucose"]).tolist() == [140, 300]
assert len(M.EVENTS) == 13

# ---- 2-3. tasks and features -------------------------------------------------------------------------
rng = np.random.default_rng(0)
X = blank(4)
for i in range(4):
    for s in M.VITALS:
        put(X, i, s, np.arange(72), rng.normal(90, 10, 72))
X[2, 2 * K["sbp"] + 1, :] = 0                                     # stay 2: SBP never measured
F = M.summary_features(X)
assert F.shape == (4, 48) and len(M.FEATURE_NAMES) == 48
j = M.FEATURE_NAMES.index("sbp_n_measured")
assert F[:, j].tolist() == [72, 72, 0, 72] and np.isnan(F[2, M.FEATURE_NAMES.index("sbp_mean")])
assert F[0, M.FEATURE_NAMES.index("glucose_n_measured")] == 0
Fc, yc, rows, _ = M.task_dataset(M.TASKS["crossing_sbp"], X)
assert len(Fc) == 4 * len(M.CROSSING_ANCHORS) and np.isnan(yc[rows == 2]).all()
Ff, yf, rf, pers = M.task_dataset(M.TASKS["forecast_heart_rate"], X)
assert Ff.shape[1] == 5 * 6 and np.allclose(pers[:4], X[:, 0, 11]) and np.allclose(yf[:4], X[:, 0, 12])

# ---- 4. statistics -----------------------------------------------------------------------------------
y = rng.integers(0, 2, 600)
a = y + rng.normal(0, 0.9, 600)
b = y + rng.normal(0, 1.2, 600)
d = M.delong_test(y, a, b)
assert abs(d["auc_a"] - roc_auc_score(y, a)) < 1e-12 and abs(d["auc_b"] - roc_auc_score(y, b)) < 1e-12
assert abs(M.fast_auc(y, a) - roc_auc_score(y, a)) < 1e-12
assert abs(M.fast_auc(y, np.round(a)) - roc_auc_score(y, np.round(a))) < 1e-12      # ties
yt = rng.integers(0, 2, 300)
at = np.round(yt + rng.normal(0, 1, 300))
assert abs(M.delong_test(yt, at, at + 0)["auc_a"] - roc_auc_score(yt, at)) < 1e-12   # ties via midranks
g = np.array([5, 5, 1, 2, 2, 2, 9])
for idx in M.bootstrap_indices(7, g, 50, 0):                      # every resample holds whole patients
    c = np.bincount(idx, minlength=7)
    assert c[0] == c[1] and c[3] == c[4] == c[5]
pt, lo, hi = M.auc_ci(y, a, None, 300)
assert lo <= pt <= hi
# patient-level CI: the same stays duplicated 4 times as one patient each -> iid bootstrap is too narrow
y4, a4, g4 = np.repeat(y, 4), np.repeat(a, 4), np.repeat(np.arange(600), 4)
_, lo_i, hi_i = M.auc_ci(y4, a4, None, 300)
_, lo_g, hi_g = M.auc_ci(y4, a4, g4, 300)
assert (hi_g - lo_g) > 1.5 * (hi_i - lo_i)
p, lo, hi = M.paired_delta_ci(y, a, b, None, 300)
assert lo <= p <= hi and p > 0                                   # a is the better score
assert M.ni_decision((0.0, -0.01, 0.015), 0.02) == "non-inferior"
assert M.ni_decision((0.03, 0.025, 0.04), 0.02) == "inferior"
assert M.ni_decision((0.02, 0.005, 0.03), 0.02) == "inconclusive"
assert M.equivalent(-0.01, 0.02, 0.05) and not M.equivalent(-0.06, 0.0, 0.05)
r = M.rate_gap(np.r_[np.ones(50), np.zeros(50)], np.r_[np.ones(50), np.zeros(50)], margin=0.05, n_boot=200)
assert r["gap"] == 0 and r["ci"][0] <= 0 <= r["ci"][1]
yy = rng.normal(0, 1, 500)
dm = M.paired_mae_delta_ci(yy, yy + 0.1, yy + 0.5, None, 200)
assert abs(dm[0] - 0.4) < 1e-9 and dm[1] > 0                     # TSTR worse -> positive delta

# ---- 6. fidelity metrics -----------------------------------------------------------------------------
xa = rng.normal(0, 1, 5000)
assert M.mdd(xa, xa) == 0 and M.mdd(xa, xa + 3) > M.mdd(xa, xa + 0.3) > 0
n, T, phi = 300, 72, 0.8
e = np.zeros((n, T)); e[:, 0] = rng.normal(0, 1, n)
for t in range(1, T):
    e[:, t] = phi * e[:, t - 1] + rng.normal(0, np.sqrt(1 - phi ** 2), n)
X = blank(n)
for i in range(n):
    put(X, i, "heart_rate", np.arange(T), 90 + 10 * e[i])
    put(X, i, "sbp", np.arange(T), 120 + 10 * e[i] + rng.normal(0, 5, T))   # correlated with HR
acf = M.acf_pooled(X, "heart_rate", (1, 2))
assert abs(acf[0] - phi) < 0.05 and abs(acf[1] - phi ** 2) < 0.07
C = M.cross_correlations(X, ["heart_rate", "sbp"])
assert C[0, 1] > 0.8
# sample-level metrics: fresh sample good, shifted bad on precision, collapsed bad on recall, copies unauthentic
R2, S2 = rng.normal(0, 1, (2000, 2)), rng.normal(0, 1, (2000, 2))
ok2 = M.alpha_beta_authenticity(R2, S2)                           # 2-D: both curves close to the diagonal
assert ok2["IP_alpha"] > 0.9 and ok2["IR_beta"] > 0.9
R = rng.normal(0, 1, (2000, 8))
fresh = rng.normal(0, 1, (2000, 8))
ok = M.alpha_beta_authenticity(R, fresh)
assert ok["IP_alpha"] > 0.85 and ok["authenticity"] > 0.3
assert ok["IR_beta"] < 0.7                                        # documented: IR is not calibrated in 8-D
assert M.alpha_beta_authenticity(R, fresh + 5)["IP_alpha"] < 0.2
collapsed = np.repeat(fresh[:20], 100, axis=0) + rng.normal(0, 0.01, (2000, 8))
assert M.alpha_beta_authenticity(R, collapsed)["recall_at_beta1"] < 0.5 * ok["recall_at_beta1"]
assert M.alpha_beta_authenticity(R, R + rng.normal(0, 1e-4, R.shape))["authenticity"] < 0.05

# ---- 8. privacy on toy cohorts -------------------------------------------------------------------------
def toy(n, seed):
    r = np.random.default_rng(seed)
    X = blank(n)
    base = r.normal(0, 1, (n, 5))
    for i, s in enumerate(M.VITALS):
        X[:, 2 * i] = [90, 120, 60, 18, 97][i] + 10 * base[:, [i]] + r.normal(0, 3, (n, 72))
        X[:, 2 * i + 1] = 1
    return M.Cohort(X, np.arange(n) + seed * 10 ** 6, name=str(seed))


tr, te = toy(800, 1), toy(800, 2)
copies = M.Cohort(tr.X + rng.normal(0, 1e-3, tr.X.shape) * np.tile([1, 0], 8)[None, :, None], name="copies")
fresh = toy(800, 3)
pc = M.privacy(tr, te, copies, n=800)
pf = M.privacy(tr, te, fresh, n=800)
for kind in ("summary", "raw_vitals"):
    assert pc[kind]["mia_auroc"] > 0.95 and pc[kind]["privacy_loss"] > 0.3 and pc[kind]["share_synth_below_holdout_p5"] > 0.9
    assert abs(pf[kind]["mia_auroc"] - 0.5) < 0.08 and abs(pf[kind]["privacy_loss"]) < 0.08
    assert pf[kind]["share_synth_below_holdout_p5"] < 0.15

# ---- 9. end-to-end runner on toy cohorts ----------------------------------------------------------------
def toy_icu(n, seed, with_labels=True):
    r = np.random.default_rng(seed)
    X = blank(n)
    mean = {"heart_rate": 90, "sbp": 115, "dbp": 60, "resp_rate": 19, "spo2": 96}
    sd = {"heart_rate": 15, "sbp": 18, "dbp": 11, "resp_rate": 5, "spo2": 2.5}
    lvl = r.normal(0, 1, (n, 1))                                   # shared severity drives several signals
    for s in M.VITALS:
        e = np.zeros((n, 72)); e[:, 0] = r.normal(0, 1, n)
        for t_ in range(1, 72):
            e[:, t_] = 0.8 * e[:, t_ - 1] + r.normal(0, 0.6, n)
        sign = -1 if s in ("sbp", "dbp", "spo2") else 1
        vals = mean[s] + sd[s] * (0.7 * e + sign * 0.7 * lvl)
        if s == "spo2":
            vals = np.minimum(vals, 100)
        X[:, 2 * K[s]], X[:, 2 * K[s] + 1] = vals, (r.random((n, 72)) > 0.04)
    for s, mu, sg in (("glucose", 140, 40), ("creatinine", 1.1, 0.4), ("sodium", 139, 4)):
        for i in range(n):
            hrs = np.sort(r.choice(72, 5, replace=False))
            drift = (0.4 * max(lvl[i, 0], 0) * np.arange(5)) if s == "creatinine" else 0
            put(X, i, s, hrs, np.maximum(mu + sg * r.normal(0, 1, 5) * (0.3 if s == "creatinine" else 1) + drift, 0.2))
    labels = {"mortality": (r.random(n) < 1 / (1 + np.exp(-(lvl[:, 0] * 1.5 - 1.6)))).astype(float)} if with_labels else {}
    groups = {"sex": r.choice(["F", "M"], n), "race_group": r.choice(["White", "Black", "Other or unknown"], n, p=[0.6, 0.2, 0.2])}
    return M.Cohort(X, np.arange(n) + seed * 10 ** 6, labels, groups, f"toy{seed}")


rt, rte, sy = toy_icu(600, 11), toy_icu(400, 12), toy_icu(600, 13)
sy.subjects, sy.groups = None, {}
small = ["hypotension_h49_72", "aki_new_h25_72", "crossing_sbp", "forecast_heart_rate", "outcome_mortality"]
ref = M._jsonable(M.real_reference(rt, rte, small, n_boot=40))
rep = M.evaluate(rt, rte, sy, None, small, n_boot=40, max_n=300, reference=ref, subgroup_keys=("sex",), verbose=False)
assert [r["task"] for r in rep["tier1"]] == small and all(r["decision"] in ("non-inferior", "inconclusive", "inferior") for r in rep["tier1"])
assert all("underpowered" in r for r in rep["tier1"]) and "subgroups" in rep["tier1"][0]
assert set(rep) >= {"meta", "tier1", "tier2", "tier3", "privacy"} and len(rep["tier3"]["events"]) == 13
assert "outcome_gradient" in rep["tier3"]                          # synthetic carries mortality labels
assert rep["meta"]["resolution_matching"] and "synthetic_raw_output" in rep["tier2"]["validity"]
md = M.report_markdown(rep)
for head in ("## Tier 1", "## Tier 2", "## Tier 3", "## Privacy"):
    assert head in md
sy_nolab = M.Cohort(sy.X, None, {}, {}, "nolabels")
rep2 = M.evaluate(rt, rte, sy_nolab, None, ["hypotension_h49_72", "outcome_mortality"], n_boot=30, max_n=300,
                  run_tier2=False, run_privacy=False, verbose=False)
assert [s_["task"] for s_ in rep2["skipped"]] == ["outcome_mortality"]   # outcome tasks need synthetic labels
json_ok = __import__("json").dumps(rep)                           # report is JSON-serializable

# ---- real-data reproduction (optional) -----------------------------------------------------------------
if "--real" in sys.argv:
    from paths import PATHS as P
    tr = M.load_real(P["train"], P["stays_train"], P["subjects_train"], P["annotations"], "real_train")
    te = M.load_real(P["test"], P["stays_test"], P["subjects_test"], P["annotations"], "real_test")
    assert len(tr) == 21860 and len(te) == 7338 and not set(tr.subjects) & set(te.subjects)
    X = np.concatenate([tr.X, te.X])
    f = {k: round(float(v.mean()), 3) for k, v in M.event_flags(X).items()}
    review = {"map_lt65_2h": 0.487, "sbp_lt90_2h": 0.304, "spo2_lt90_2h": 0.087, "hr_gt100_2h": 0.595, "hr_lt50_2h": 0.059,
              "shock_index_ge1_2h": 0.442, "aki_creatinine": 0.336, "glucose_lt70_any": 0.068, "glucose_lt54_any": 0.019,
              "glucose_gt180_any": 0.388, "sodium_lt135_any": 0.373, "sodium_lt125_any": 0.025, "sodium_gt145_any": 0.179}
    assert f == review, f
    v, m = M.split(X)
    assert round(M.aki_creatinine(v, m, "first").mean(), 3) == 0.261 and round(M.aki_creatinine(v, m, "minimum").mean(), 3) == 0.529
    assert [round(float((M.funk_grade(M.first_value(v, m, "sodium")) == i).mean()), 3) for i in range(7)] == \
        [0.015, 0.039, 0.14, 0.735, 0.053, 0.009, 0.006]
    assert round(float(np.nanmean(tr.labels["mortality"])), 3) == 0.179 or round(float(np.nanmean(np.r_[tr.labels["mortality"], te.labels["mortality"]])), 3) == 0.179
    for name, lo_, hi_ in (("outcome_mortality", 0.76, 0.78), ("hypotension_h49_72", 0.83, 0.845), ("aki_new_h25_72", 0.74, 0.76)):
        task = M.TASKS[name]
        Fr, yr, _, _ = M.task_dataset(task, tr.X, tr.labels)
        Ft, yt_, _, _ = M.task_dataset(task, te.X, te.labels)
        k = np.isfinite(yt_)
        auc = roc_auc_score(yt_[k], M.fit_predict(Fr, yr, Ft[k]))
        assert lo_ <= auc <= hi_, (name, auc)
    print("real-data checks passed: event rates, AKI baselines and sodium grades match the review; baseline AUCs in range")

print("all assertions passed")
