"""
Utility evaluation for synthetic versions of our preprocessed MIMIC-IV ICU tensor.

Data contract (channels.json): each stay is a 16 x 72 array, 8 signals each followed by its mask channel
(heart_rate, sbp, dbp, resp_rate, spo2, glucose, creatinine, sodium), 1-hour slots over the first 72 hours
of the ICU stay. mask == 1 marks a real measurement; other slots hold forward-filled values, a fixed
leading fill, and every value channel is clipped after imputation. All metrics below therefore use
measured slots only (values[mask == 1]); a generator's values in unmeasured slots are ignored.

Sections
    0. I/O and validation (real cohort from the uploaded files, synthetic arrays from any generator)
    1. Rule-based labels computed from the series itself (same code on real and synthetic stays)
    2. Task registry (input window, label rule, source)
    3. Features and models (per-window summaries + gradient boosting; replaceable)
    4. Statistics (patient-level bootstrap, DeLong, non-inferiority and equivalence decisions)
    5. Tier 1  task utility: TSTR vs TRTR (rule-based, threshold-crossing, forecasting, outcome, subgroups)
    6. Tier 2  fidelity on measured slots, each metric next to a real-vs-real reference
    7. Tier 3  clinical-event consistency (overall and within outcome groups)
    8. Privacy (reported separately from utility): DCR, distance-based membership inference, NNAA
    9. Runner, Markdown report and command line

Design choices (all configurable)
  - Synthetic data are generated from the TRAINING split only; the real TEST split is never shown to the
    generator. TRTR trains on real train, TSTR on synthetic, both are tested on real test.
  - Training sets are size-matched (both subsampled to the smaller of the two) so that TSTR is not
    penalized or favored by sample size.
  - Confidence intervals resample PATIENTS (one-third of stays are repeat stays of the same patient).
  - Non-inferiority margin: pre-registered, 0.02 AUROC by default (5% of TRTR MAE for forecasting). Each task
    also reports a real-vs-real reference: a model trained on held-out real patients in place of synthetic
    data, i.e. what a perfect generator would score. Tasks where even that reference cannot pass are flagged
    as underpowered. The seed and model-family spread of real models (the anchor in the ECG review) is
    reported next to it.
  - Every Tier 2 and Tier 3 metric is also computed for real test vs real train. That reference is what a
    perfect generator (a fresh sample of patients from the same population) would score.
  - Resolution matching (default on): synthetic values are snapped to the grid of values observed in the
    real training data before labels and metrics, because real values are mostly integers and threshold
    rules are sensitive to that (see ValueGrid). Validity is also reported on the raw output.

Sources for definitions: KDIGO (2012) creatinine criteria; Evans et al. (2021) MAP 65 mmHg;
Koch et al. (2019) shock index; American Diabetes Association (2025) glucose levels; Spasovski et al.
(2014) and Funk et al. (2010) sodium grades; Egi et al. (2006) glucose variability; Esteban et al. (2017)
TSTR and next-hour threshold crossing; L. P. Zimmerman et al. (2019) day-1 AKI prediction;
Yoon et al. (2019) discriminative and predictive scores; Ang et al. (2023) TSGBench feature-based measures;
Alaa et al. (2022) alpha-precision, beta-recall, authenticity; Chen et al. (2020) distance-based
membership inference; Yale et al. (2019) nearest-neighbor adversarial accuracy.

Status: reference implementation, checked by test_mimic_utility.py and by positive/negative control
generators (validate_pipeline.py). Not a substitute for pre-registering tasks and margins.
Dependencies: numpy, scipy, scikit-learn (tested with numpy 2.x, scikit-learn 1.x).
Threads: scikit-learn's gradient boosting uses OpenMP. When several evaluations run at once on one machine
(e.g. one job per generator on a cluster node), set OMP_NUM_THREADS so the total does not exceed the cores;
oversubscription made a 35-second test run for more than 20 minutes in our checks.
"""
from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# =====================================================================================
# 0. I/O and validation
# =====================================================================================
SIG: List[str] = ["heart_rate", "sbp", "dbp", "resp_rate", "spo2", "glucose", "creatinine", "sodium"]
VITALS: List[str] = SIG[:5]
LABS: List[str] = SIG[5:]
N_CH, N_T = 16, 72
CHANNELS: List[str] = [c for s in SIG for c in (s, s + "_mask")]
# Defaults copied from channels.json of our preprocessing; load_channels() reads the real file.
DEFAULT_LEADING_FILL = {"heart_rate": 85.0, "sbp": 115.5, "dbp": 60.0, "resp_rate": 19.0, "spo2": 97.0,
                        "glucose": 130.0, "creatinine": 1.100000023841858, "sodium": 139.0}
DEFAULT_CLIP = {"heart_rate": (20.0, 300.0), "sbp": (40.0, 300.0), "dbp": (20.0, 200.0),
                "resp_rate": (4.0, 120.0), "spo2": (50.0, 100.0), "glucose": (10.0, 1000.0),
                "creatinine": (0.1, 15.0), "sodium": (110.0, 165.0)}


def load_channels(path: Optional[str]) -> Dict:
    """Read channels.json and check that the channel order is the one this module assumes."""
    if path is None:
        return {"channels": CHANNELS, "leading_fill": DEFAULT_LEADING_FILL, "clip_bounds": DEFAULT_CLIP}
    with open(path) as f:
        meta = json.load(f)
    if meta.get("channels") != CHANNELS:
        raise ValueError(f"channel order differs from {CHANNELS}; reorder the arrays before evaluating")
    clip = {s: (b["low"], b["high"]) for s, b in meta.get("clip_bounds", {}).items()} or DEFAULT_CLIP
    return {"channels": meta["channels"], "leading_fill": meta.get("leading_fill", DEFAULT_LEADING_FILL),
            "clip_bounds": clip, "n_steps": meta.get("n_steps", N_T)}


def split(X: np.ndarray) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Values and boolean masks keyed by signal name. Masks are thresholded at 0.5, so continuous mask
    outputs of a generator are binarized the same way everywhere."""
    X = np.asarray(X)
    return ({s: X[:, 2 * i, :] for i, s in enumerate(SIG)},
            {s: X[:, 2 * i + 1, :] > 0.5 for i, s in enumerate(SIG)})


def prepare_synthetic(X: np.ndarray, mask_threshold: float = 0.5) -> Tuple[np.ndarray, List[str]]:
    """Bring a generator's output into the real layout (stays x 16 x 72, masks in {0, 1}).

    Accepts stays x 16 x 72, stays x 72 x 16, or values only (stays x 8 x 72 or stays x 72 x 8).
    Values-only output gets all-ones masks and a warning: missingness tests are then meaningless and the
    sparse laboratory signals are treated as hourly measurements. Values must already be in original units
    (de-normalized); a range check flags outputs that look normalized."""
    notes: List[str] = []
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 3:
        raise ValueError(f"expected a 3-D array, got shape {X.shape}")
    if X.shape[1:] == (N_T, N_CH) or X.shape[1:] == (N_T, len(SIG)):
        X = X.transpose(0, 2, 1)
        notes.append("transposed from stays x time x channels")
    if X.shape[1:] == (len(SIG), N_T):
        notes.append("values only: masks set to 1 everywhere; missingness results are not interpretable")
        full = np.empty((X.shape[0], N_CH, N_T))
        full[:, 0::2], full[:, 1::2] = X, 1.0
        X = full
    if X.shape[1:] != (N_CH, N_T):
        raise ValueError(f"expected stays x {N_CH} x {N_T}, got {X.shape}")
    bad = ~np.isfinite(X)
    if bad.any():
        notes.append(f"{int(bad.sum())} non-finite values set to 0 and their slots marked unmeasured")
        X = X.copy()
        for i in range(len(SIG)):
            b = bad[:, 2 * i] | bad[:, 2 * i + 1]
            X[:, 2 * i][b], X[:, 2 * i + 1][b] = 0.0, 0.0
    X = X.copy()
    X[:, 1::2] = (X[:, 1::2] > mask_threshold).astype(np.float64)
    v, m = split(X)
    hr = v["heart_rate"][m["heart_rate"]]
    if hr.size and not 30 <= np.median(hr) <= 200:
        notes.append(f"median measured heart rate {np.median(hr):.2f}: values may still be normalized")
    return X, notes


class ValueGrid:
    """Resolution matching. Real measured values sit on a discrete grid: 92-99.9% of measured vital signs,
    glucose and sodium values in our training data are integers, creatinine has 0.1 mg/dL steps, and 5.9%
    of sodium values are exactly 135 mmol/L. Generators output continuous values, so threshold rules such as
    sodium < 135 would count values like 134.97 that real data never contain. Adding 0.001 noise to the
    real data shifted event rates by up to 5 percentage points. snap() moves each synthetic value to the
    nearest value observed in the REAL TRAINING data for that signal. Values outside the observed range are
    left unchanged, so range and validity checks still see them."""

    def __init__(self, X_real: np.ndarray):
        v, m = split(X_real)
        self.grid = {s: np.unique(v[s][m[s]]) for s in SIG}

    def snap(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        X = np.array(X, dtype=np.float64, copy=True)
        _, m = split(X)
        shift = {}
        for i, s in enumerate(SIG):
            g, v = self.grid[s], X[:, 2 * i]
            if len(g) < 2:
                continue
            idx = np.clip(np.searchsorted(g, v), 1, len(g) - 1)
            lo, hi = g[idx - 1], g[idx]
            snapped = np.where(np.abs(v - lo) <= np.abs(hi - v), lo, hi)
            new = np.where((v >= g[0]) & (v <= g[-1]), snapped, v)
            shift[s] = float(np.abs(new - v)[m[s]].mean()) if m[s].any() else 0.0
            X[:, 2 * i] = new
        return X, shift


def race_group(r: Optional[str]) -> str:
    r = r or ""
    for prefix, g in (("WHITE", "White"), ("BLACK", "Black"), ("HISPANIC", "Hispanic/Latino"), ("ASIAN", "Asian")):
        if r.startswith(prefix):
            return g
    return "Other or unknown"


def insurance_group(x: Optional[str]) -> str:
    return x if x in ("Medicare", "Medicaid", "Private") else "Other or missing"


def age_band(a: Optional[float]) -> str:
    if a is None or not np.isfinite(a):
        return "unknown"
    return "<50" if a < 50 else "50-64" if a < 65 else "65-79" if a < 80 else ">=80"


# ICD prefixes used in the review (Section 4.5); replace with a published mapping for the paper.
ICD_PHENOTYPES = {
    "icd_sepsis": (("038", "99591", "99592", "78552"), ("A40", "A41", "R652")),
    "icd_heart_failure": (("428",), ("I50",)),
    "icd_aki": (("584",), ("N17",)),
}
HOME = {"HOME", "HOME HEALTH CARE"}


@dataclass
class Cohort:
    """One set of stays. `labels` holds outcome labels (real: annotations; synthetic: only if the generator
    is conditioned on them), float arrays with NaN = not applicable. `groups` holds subgroup attributes."""
    X: np.ndarray
    subjects: Optional[np.ndarray] = None
    labels: Dict[str, np.ndarray] = field(default_factory=dict)
    groups: Dict[str, np.ndarray] = field(default_factory=dict)
    name: str = ""

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.float64)
        if self.X.ndim != 3 or self.X.shape[1:] != (N_CH, N_T):
            raise ValueError(f"{self.name}: expected stays x {N_CH} x {N_T}, got {self.X.shape}")
        n = len(self.X)
        if self.subjects is not None and len(self.subjects) != n:
            raise ValueError(f"{self.name}: {len(self.subjects)} subject ids for {n} stays")
        for k, a in {**self.labels, **self.groups}.items():
            if len(a) != n:
                raise ValueError(f"{self.name}: '{k}' has {len(a)} entries for {n} stays")

    def __len__(self):
        return len(self.X)

    def subset(self, idx: np.ndarray, name: Optional[str] = None) -> "Cohort":
        return Cohort(self.X[idx], None if self.subjects is None else self.subjects[idx],
                      {k: a[idx] for k, a in self.labels.items()}, {k: a[idx] for k, a in self.groups.items()},
                      name or self.name)


def labels_from_annotations(stays: Sequence, ann: Dict) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Outcome labels and subgroup attributes for real stays, from annotations.json['stays']."""
    rows = [ann[str(int(s))] for s in stays]
    f = lambda key: np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])
    mort = f("mortality")
    readm = f("readmission_30day")
    readm[mort == 1] = np.nan                      # 0 for in-hospital deaths in the file: not applicable
    los = f("los_icu_days")
    home = np.array([np.nan if (r.get("mortality") == 1 or r.get("discharge_location") in (None, "DIED", "AGAINST ADVICE"))
                     else float(r["discharge_location"] in HOME) for r in rows])
    labels = {"mortality": mort, "mortality_in_icu": f("mortality_in_icu"), "mortality_30day": f("mortality_30day"),
              "mortality_1year": f("mortality_1year"), "readmission_30day": readm,
              "los_icu_gt7": np.where(np.isfinite(los), (los > 7).astype(float), np.nan), "discharge_home": home}
    for name, (p9, p10) in ICD_PHENOTYPES.items():
        out = []
        for r in rows:
            pre = p9 if r.get("icd_version") == 9 else p10 if r.get("icd_version") == 10 else p9 + p10
            out.append(float(any(str(c).startswith(pre) for c in (r.get("icd_codes") or []))))
        labels[name] = np.array(out)
    groups = {"race_group": np.array([race_group(r.get("race")) for r in rows]),
              "insurance_group": np.array([insurance_group(r.get("insurance")) for r in rows]),
              "sex": np.array([str(r.get("gender")) for r in rows]),
              "age_band": np.array([age_band(r.get("age")) for r in rows])}
    return labels, groups


def load_real(tensor_path: str, stays_path: str, subjects_path: str, annotations_path: str, name: str) -> Cohort:
    X = np.load(tensor_path)
    stays, subjects = np.load(stays_path), np.load(subjects_path)
    with open(annotations_path) as f:
        ann = json.load(f)["stays"]
    labels, groups = labels_from_annotations(stays, ann)
    return Cohort(X, subjects, labels, groups, name)


def load_synthetic(path: str, labels_path: Optional[str] = None, name: str = "synthetic") -> Tuple[Cohort, List[str]]:
    """Synthetic tensor (.npy, or .npz with key 'X') plus optional labels/attributes (.npz; numeric keys go to
    labels, string keys such as race_group go to groups)."""
    arr = np.load(path)
    X = arr["X"] if hasattr(arr, "files") else arr
    X, notes = prepare_synthetic(X)
    labels, groups = {}, {}
    if labels_path:
        z = np.load(labels_path, allow_pickle=False)
        for k in z.files:
            (groups if z[k].dtype.kind in "US" else labels)[k] = z[k].astype(str if z[k].dtype.kind in "US" else float)
    return Cohort(X, None, labels, groups, name), notes


# =====================================================================================
# 1. Rule-based labels (computed from the series; measured slots only)
# =====================================================================================
def longest_run(cond: np.ndarray) -> np.ndarray:
    """Longest run of consecutive True hours per stay."""
    run = np.zeros(cond.shape[0], int)
    best = run.copy()
    for t in range(cond.shape[1]):
        run = np.where(cond[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return best


OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal}


def sustained(v, m, sig: str, op: str, thr: float, hours: int = 2) -> np.ndarray:
    """Signal beyond threshold for >= `hours` consecutive MEASURED hours (an unmeasured hour breaks the run)."""
    return longest_run(OPS[op](v[sig], thr) & m[sig]) >= hours


def derived_map(v) -> np.ndarray:
    """MAP is not in the tensor; the standard approximation (SBP + 2 x DBP) / 3."""
    return (v["sbp"] + 2.0 * v["dbp"]) / 3.0


def hypotension(v, m, thr: float = 65.0, hours: int = 2) -> np.ndarray:
    """Derived MAP < thr mmHg for >= hours consecutive hours with SBP and DBP both measured."""
    return longest_run((derived_map(v) < thr) & m["sbp"] & m["dbp"]) >= hours


def shock_index_event(v, m, thr: float = 1.0, hours: int = 2) -> np.ndarray:
    """Shock index (HR / SBP) >= thr for >= hours consecutive hours with both measured (Koch et al., 2019)."""
    both = m["heart_rate"] & m["sbp"]
    si = np.where(both, v["heart_rate"] / np.maximum(v["sbp"], 1.0), 0.0)
    return longest_run((si >= thr) & both) >= hours


def any_measured(v, m, sig: str, op: str, thr: float) -> np.ndarray:
    """Any measured value beyond threshold (used for sparse laboratory values)."""
    return (OPS[op](v[sig], thr) & m[sig]).any(axis=1)


def aki_creatinine(v, m, baseline: str = "rolling") -> np.ndarray:
    """KDIGO serum-creatinine criteria inside the window: rise >= 0.3 mg/dL within 48 h, or >= 1.5 x baseline.
    baseline = 'rolling' (lowest earlier value; the review's default), 'first' (first measured value) or
    'minimum' (minimum of the whole window regardless of timing; retrospective, over-counts recovery)."""
    if baseline not in ("rolling", "first", "minimum"):
        raise ValueError(baseline)
    out = np.zeros(v["creatinine"].shape[0], bool)
    for i in range(len(out)):
        t = np.flatnonzero(m["creatinine"][i])
        c = v["creatinine"][i, t]
        if len(t) < 2:
            continue
        if baseline == "first":
            out[i] = bool(((c[1:] - c[0] >= 0.3) & (t[1:] - t[0] <= 48)).any() or (c[1:] >= 1.5 * c[0]).any())
        elif baseline == "minimum":
            k = int(np.argmin(c))
            out[i] = bool(((c - c[k] >= 0.3) & (np.abs(t - t[k]) <= 48)).any() or (c >= 1.5 * c[k]).any())
        else:
            for j in range(1, len(t)):
                recent = c[:j][t[:j] >= t[j] - 48]
                if (recent.size and c[j] - recent.min() >= 0.3) or c[j] >= 1.5 * c[:j].min():
                    out[i] = True
                    break
    return out


def aki_onset(v, m) -> np.ndarray:
    """Hour of the first measurement that meets the rolling-baseline KDIGO criteria; -1 if none."""
    on = np.full(v["creatinine"].shape[0], -1)
    for i in range(len(on)):
        t = np.flatnonzero(m["creatinine"][i])
        c = v["creatinine"][i, t]
        for j in range(1, len(t)):
            recent = c[:j][t[:j] >= t[j] - 48]
            if (recent.size and c[j] - recent.min() >= 0.3) or c[j] >= 1.5 * c[:j].min():
                on[i] = t[j]
                break
    return on


def first_value(v, m, sig: str) -> np.ndarray:
    """First measured value per stay (NaN if never measured)."""
    has, idx = m[sig].any(axis=1), m[sig].argmax(axis=1)
    return np.where(has, v[sig][np.arange(len(idx)), idx], np.nan)


def last_measured(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Last measured value per stay (NaN if never measured)."""
    T = mask.shape[1]
    has = mask.any(axis=1)
    idx = T - 1 - mask[:, ::-1].argmax(axis=1)
    return np.where(has, values[np.arange(len(idx)), idx], np.nan)


FUNK_GRADES = ["severe hypo (<125)", "mild hypo (125-<130)", "borderline hypo (130-<135)", "normal (135-145)",
               "borderline hyper (>145-150)", "mild hyper (>150-155)", "severe hyper (>155)"]


def funk_grade(na: np.ndarray) -> np.ndarray:
    """Dysnatremia grade index into FUNK_GRADES (Funk et al., 2010); -1 if missing."""
    na = np.asarray(na, float)
    g = np.full(na.shape, -1)
    g[na < 125] = 0
    g[(na >= 125) & (na < 130)] = 1
    g[(na >= 130) & (na < 135)] = 2
    g[(na >= 135) & (na <= 145)] = 3
    g[(na > 145) & (na <= 150)] = 4
    g[(na > 150) & (na <= 155)] = 5
    g[na > 155] = 6
    return g


def glucose_sd(v, m, min_n: int = 3) -> np.ndarray:
    """Within-stay SD of measured glucose (ddof = 1); NaN with fewer than min_n values (Egi et al., 2006)."""
    x = np.where(m["glucose"], v["glucose"], np.nan)
    n = m["glucose"].sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sd = np.nanstd(x, axis=1, ddof=1)
    return np.where(n >= min_n, sd, np.nan)


# Events over hours 0-72 for Tier 3 (rates in the review, Sections 4.2-4.4).
EVENTS: Dict[str, Callable] = {
    "map_lt65_2h": lambda v, m: hypotension(v, m),
    "sbp_lt90_2h": lambda v, m: sustained(v, m, "sbp", "<", 90),
    "spo2_lt90_2h": lambda v, m: sustained(v, m, "spo2", "<", 90),
    "hr_gt100_2h": lambda v, m: sustained(v, m, "heart_rate", ">", 100),
    "hr_lt50_2h": lambda v, m: sustained(v, m, "heart_rate", "<", 50),
    "shock_index_ge1_2h": lambda v, m: shock_index_event(v, m),
    "aki_creatinine": lambda v, m: aki_creatinine(v, m),
    "glucose_lt70_any": lambda v, m: any_measured(v, m, "glucose", "<", 70),
    "glucose_lt54_any": lambda v, m: any_measured(v, m, "glucose", "<", 54),
    "glucose_gt180_any": lambda v, m: any_measured(v, m, "glucose", ">", 180),
    "sodium_lt135_any": lambda v, m: any_measured(v, m, "sodium", "<", 135),
    "sodium_lt125_any": lambda v, m: any_measured(v, m, "sodium", "<", 125),
    "sodium_gt145_any": lambda v, m: any_measured(v, m, "sodium", ">", 145),
}


def event_flags(X: np.ndarray, events: Dict[str, Callable] = EVENTS) -> Dict[str, np.ndarray]:
    v, m = split(X)
    return {k: np.asarray(fn(v, m), bool) for k, fn in events.items()}


# =====================================================================================
# 2. Task registry
# =====================================================================================
@dataclass(frozen=True)
class Task:
    """kind: 'rule' (label from the series, one row per stay), 'crossing' (next-hour threshold crossing,
    several rows per stay), 'forecast' (next-hour value regression), 'outcome' (label from annotations;
    needs a conditional generator)."""
    name: str
    kind: str
    description: str
    input_hours: Tuple[int, int] = (0, 72)
    label_fn: Optional[Callable[[np.ndarray], np.ndarray]] = None
    outcome_key: Optional[str] = None
    signal: Optional[str] = None
    op: Optional[str] = None
    thr: Optional[float] = None
    anchors: Tuple[int, ...] = ()
    history: int = 0
    source: str = ""


def _future(rule: Callable, start: int = 48) -> Callable[[np.ndarray], np.ndarray]:
    def f(X):
        v, m = split(X[:, :, start:])
        return np.asarray(rule(v, m), float)
    return f


def _aki_new(cut: int = 24) -> Callable[[np.ndarray], np.ndarray]:
    def f(X):
        on = aki_onset(*split(X))
        y = (on > cut).astype(float)
        y[(on >= 0) & (on <= cut)] = np.nan        # AKI already in the input window: excluded
        return y
    return f


CROSSING_ANCHORS = (24, 36, 48, 60)
FORECAST_ANCHORS = (12, 24, 36, 48, 60, 71)

TASKS: Dict[str, Task] = {
    "hypotension_h49_72": Task("hypotension_h49_72", "rule", "Derived MAP < 65 mmHg for >= 2 h in hours 49-72",
                               (0, 48), _future(lambda v, m: hypotension(v, m)), source="Evans et al., 2021"),
    "shock_index_h49_72": Task("shock_index_h49_72", "rule", "Shock index >= 1.0 for >= 2 h in hours 49-72",
                               (0, 48), _future(lambda v, m: shock_index_event(v, m)), source="Koch et al., 2019"),
    "spo2_lt90_h49_72": Task("spo2_lt90_h49_72", "rule", "SpO2 < 90% for >= 2 h in hours 49-72",
                             (0, 48), _future(lambda v, m: sustained(v, m, "spo2", "<", 90))),
    "tachycardia_h49_72": Task("tachycardia_h49_72", "rule", "Heart rate > 100 bpm for >= 2 h in hours 49-72",
                               (0, 48), _future(lambda v, m: sustained(v, m, "heart_rate", ">", 100))),
    "aki_new_h25_72": Task("aki_new_h25_72", "rule", "New creatinine-criterion AKI in hours 25-72 (AKI in 0-24 excluded)",
                           (0, 24), _aki_new(24), source="KDIGO, 2012; L. P. Zimmerman et al., 2019"),
}
for _sig, _op, _thr in (("heart_rate", ">", 100.0), ("sbp", "<", 90.0), ("resp_rate", ">", 24.0), ("spo2", "<", 92.0)):
    TASKS[f"crossing_{_sig}"] = Task(f"crossing_{_sig}", "crossing",
                                     f"{_sig} {_op} {_thr:g} in the next hour, from the previous 12 h",
                                     signal=_sig, op=_op, thr=_thr, anchors=CROSSING_ANCHORS, history=12,
                                     source="in the spirit of Esteban et al., 2017 (thresholds are ours)")
for _sig in VITALS:
    TASKS[f"forecast_{_sig}"] = Task(f"forecast_{_sig}", "forecast", f"{_sig} value in the next hour from the previous 6 h",
                                     signal=_sig, anchors=FORECAST_ANCHORS, history=6,
                                     source="predictive score of Yoon et al., 2019, one-step version")
for _key, _desc in (("mortality", "In-hospital mortality"), ("mortality_in_icu", "ICU mortality"),
                    ("mortality_1year", "1-year mortality"), ("los_icu_gt7", "ICU length of stay > 7 days"),
                    ("discharge_home", "Discharge home (survivors)"), ("icd_sepsis", "ICD-coded sepsis"),
                    ("icd_heart_failure", "ICD-coded heart failure"), ("icd_aki", "ICD-coded AKI")):
    TASKS[f"outcome_{_key}"] = Task(f"outcome_{_key}", "outcome", _desc + " from hours 0-72", outcome_key=_key,
                                    source="needs a generator conditioned on this label")

DEFAULT_TASKS = ["hypotension_h49_72", "shock_index_h49_72", "spo2_lt90_h49_72", "tachycardia_h49_72",
                 "aki_new_h25_72", "crossing_heart_rate", "crossing_sbp", "crossing_resp_rate", "crossing_spo2",
                 "forecast_heart_rate", "forecast_sbp", "forecast_spo2"] + \
                [k for k in TASKS if k.startswith("outcome_")]


# =====================================================================================
# 3. Features and models
# =====================================================================================
FEATURE_STATS = ("mean", "min", "max", "std", "last", "n_measured")
FEATURE_NAMES = [f"{s}_{k}" for s in SIG for k in FEATURE_STATS]


def summary_features(X: np.ndarray, start: int = 0, end: Optional[int] = None) -> np.ndarray:
    """Per signal over measured slots of the window: mean, min, max, SD, last measured value, number of
    measured slots (missingness is informative; Che et al., 2018). NaN where nothing was measured."""
    v, m = split(np.asarray(X)[:, :, start:end])
    cols = []
    for s in SIG:
        x = np.where(m[s], v[s], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            cols += [np.nanmean(x, 1), np.nanmin(x, 1), np.nanmax(x, 1), np.nanstd(x, 1)]
        cols += [last_measured(v[s], m[s]), m[s].sum(1).astype(float)]
    return np.column_stack(cols)


def make_classifier(kind: str = "hgb", seed: int = 0):
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, random_state=seed)
    if kind == "lr":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000, C=1.0, random_state=seed))
    raise ValueError(kind)


def fit_predict(F_tr, y_tr, F_te, kind: str = "hgb", seed: int = 0) -> np.ndarray:
    keep = np.isfinite(y_tr)
    if kind == "lr":
        F_tr = np.where(np.isfinite(F_tr), F_tr, np.nan)
        all_nan = np.isnan(F_tr[keep]).all(axis=0)     # columns never observed in training
        F_tr, F_te = F_tr[:, ~all_nan], F_te[:, ~all_nan]
    y = y_tr[keep].astype(int)
    if y.min() == y.max():
        return np.full(len(F_te), float(y[0]))
    clf = make_classifier(kind, seed).fit(F_tr[keep], y)
    return clf.predict_proba(F_te)[:, 1]


def task_dataset(task: Task, X: np.ndarray, labels: Optional[Dict[str, np.ndarray]] = None
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Features, label (NaN = excluded), row-to-stay index, and (forecast only) persistence prediction."""
    n = len(X)
    if task.kind == "rule":
        F = summary_features(X, *task.input_hours)
        return F, task.label_fn(X), np.arange(n), None
    if task.kind == "outcome":
        if not labels or task.outcome_key not in labels:
            raise KeyError(task.outcome_key)
        return summary_features(X, *task.input_hours), np.asarray(labels[task.outcome_key], float), np.arange(n), None
    v, m = split(X)
    Fs, ys, rows, pers = [], [], [], []
    for t in task.anchors:
        if task.kind == "crossing":
            F = summary_features(X, t - task.history, t)
            y = np.where(m[task.signal][:, t], OPS[task.op](v[task.signal][:, t], task.thr).astype(float), np.nan)
            p = None
        else:  # forecast: raw values of the 5 vitals over the history (NaN where unmeasured)
            F = np.column_stack([np.where(m[s][:, t - task.history:t], v[s][:, t - task.history:t], np.nan) for s in VITALS])
            last = last_measured(v[task.signal][:, t - task.history:t], m[task.signal][:, t - task.history:t])
            y = np.where(m[task.signal][:, t] & np.isfinite(last), v[task.signal][:, t], np.nan)
            p = last
        Fs.append(F); ys.append(y); rows.append(np.arange(n))
        if p is not None:
            pers.append(p)
    return np.vstack(Fs), np.concatenate(ys), np.concatenate(rows), (np.concatenate(pers) if pers else None)


# =====================================================================================
# 4. Statistics
# =====================================================================================
def _cluster_layout(groups: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stay indices sorted by patient, and each patient's start position and number of stays."""
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    order = np.argsort(inv, kind="stable")
    counts = np.bincount(inv)
    starts = np.r_[0, np.cumsum(counts)[:-1]]
    return order, starts, counts


def bootstrap_indices(n: int, groups: Optional[np.ndarray] = None, n_boot: int = 1000, seed: int = 0) -> Iterator[np.ndarray]:
    """Percentile-bootstrap resamples; with `groups` (patient ids) whole patients are resampled."""
    rng = np.random.default_rng(seed)
    if groups is None:
        for _ in range(n_boot):
            yield rng.integers(0, n, n)
        return
    order, starts, counts = _cluster_layout(groups)
    k = len(counts)
    for _ in range(n_boot):
        pick = rng.integers(0, k, k)
        lens, st = counts[pick], starts[pick]
        offs = np.repeat(st - np.r_[0, np.cumsum(lens)[:-1]], lens)
        yield order[np.arange(lens.sum()) + offs]


def fast_auc(y, s) -> float:
    """AUROC via the Mann-Whitney rank sum (ties get average ranks); identical to sklearn's roc_auc_score for
    binary labels but without its input validation, which dominates the cost inside bootstrap loops."""
    y = np.asarray(y).astype(bool)
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    r = stats.rankdata(s)
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def auc_ci(y, s, groups=None, n_boot=1000, alpha=0.05, seed=0, metric=fast_auc) -> Tuple[float, float, float]:
    y, s = np.asarray(y), np.asarray(s)
    vals = [metric(y[i], s[i]) for i in bootstrap_indices(len(y), groups, n_boot, seed) if 0 < y[i].sum() < len(i)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, s)), float(lo), float(hi)


def paired_delta_ci(y, s_ref, s_new, groups=None, n_boot=1000, alpha=0.05, seed=0, metric=fast_auc
                    ) -> Tuple[float, float, float]:
    """Delta = metric(ref) - metric(new) (e.g. AUC(TRTR) - AUC(TSTR)) with a paired bootstrap: the same
    resampled test patients score both models."""
    y, s_ref, s_new = map(np.asarray, (y, s_ref, s_new))
    vals = [metric(y[i], s_ref[i]) - metric(y[i], s_new[i])
            for i in bootstrap_indices(len(y), groups, n_boot, seed) if 0 < y[i].sum() < len(i)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(metric(y, s_ref) - metric(y, s_new)), float(lo), float(hi)


def paired_mae_delta_ci(y, p_ref, p_new, groups=None, n_boot=1000, alpha=0.05, seed=0) -> Tuple[float, float, float]:
    """Delta = MAE(new) - MAE(ref); positive means the new (TSTR) model is worse."""
    e_ref, e_new = np.abs(np.asarray(y) - p_ref), np.abs(np.asarray(y) - p_new)
    d = e_new - e_ref
    vals = [d[i].mean() for i in bootstrap_indices(len(d), groups, n_boot, seed)]
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(d.mean()), float(lo), float(hi)


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(z):
        j = i
        while j < len(z) and z[j] == z[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(x))
    out[order] = ranks
    return out


def delong_test(y_true, score_a, score_b) -> Dict[str, float]:
    """DeLong et al. (1988) test for two correlated AUCs on the same binary labels (assumes independent
    rows, so with repeat stays it is descriptive; the patient bootstrap is the decision statistic)."""
    y_true = np.asarray(y_true).astype(bool)
    pos, neg = y_true, ~y_true
    m, n = int(pos.sum()), int(neg.sum())
    preds = np.vstack([score_a, score_b]).astype(float)
    tx = np.array([_midrank(p[pos]) for p in preds])
    ty = np.array([_midrank(p[neg]) for p in preds])
    tz = np.array([_midrank(np.concatenate([p[pos], p[neg]])) for p in preds])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v10 = (tz[:, :m] - tx) / n
    v01 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v10) / m + np.cov(v01) / n
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (aucs[0] - aucs[1]) / np.sqrt(var) if var > 0 else 0.0
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}


def non_inferior(delta_ci_upper: float, margin: float) -> bool:
    """Non-inferior if the upper CI bound of (reference - new) is below the margin."""
    return bool(delta_ci_upper < margin)


def ni_decision(delta: Sequence[float], margin: float) -> str:
    """Three-way reading of the CI of delta = reference - new against the margin:
    'non-inferior' (upper bound < margin), 'inferior' (lower bound > margin), else 'inconclusive'."""
    lo, hi = delta[1], delta[2]
    return "non-inferior" if hi < margin else "inferior" if lo > margin else "inconclusive"


def equivalent(lo: float, hi: float, margin: float) -> bool:
    """Equivalent if the whole CI lies within +/- margin."""
    return bool(-margin <= lo and hi <= margin)


def rate_gap(real_flags, syn_flags, real_groups=None, margin: Optional[float] = None, n_boot=1000, alpha=0.05,
             seed=0) -> Dict:
    """Rate difference (synthetic - real) with a bootstrap CI: real stays resampled by patient, synthetic
    stays independently."""
    a, b = np.asarray(real_flags, float), np.asarray(syn_flags, float)
    ia = bootstrap_indices(len(a), real_groups, n_boot, seed)
    ib = bootstrap_indices(len(b), None, n_boot, seed + 1)
    d = [b[j].mean() - a[i].mean() for i, j in zip(ia, ib)]
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    res = {"rate_real": float(a.mean()), "rate_synth": float(b.mean()), "gap": float(b.mean() - a.mean()),
           "ci": (float(lo), float(hi)), "n_real": len(a), "n_synth": len(b)}
    if margin is not None:
        res["equivalent"] = equivalent(lo, hi, margin)
    return res


# =====================================================================================
# 5. Tier 1: task utility (TSTR vs TRTR)
# =====================================================================================
def _size_match(n_a: int, n_b: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    k = min(n_a, n_b)
    return np.sort(rng.choice(n_a, k, replace=False)), np.sort(rng.choice(n_b, k, replace=False))


def _patient_groups(cohort: Cohort, rows: np.ndarray) -> Optional[np.ndarray]:
    return None if cohort.subjects is None else cohort.subjects[rows]


def tier1_task(task: Task, real_train: Cohort, real_test: Cohort, synth: Cohort, margin: float,
               n_boot: int = 1000, seed: int = 0, model: str = "hgb", size_match: bool = True,
               subgroup_keys: Sequence[str] = (), min_group: int = 200, min_pos: int = 20) -> Dict:
    """TRTR vs TSTR for one classification task, tested on real test stays."""
    ia, ib = _size_match(len(real_train), len(synth), seed) if size_match else (np.arange(len(real_train)), np.arange(len(synth)))
    rt, sy = real_train.subset(ia), synth.subset(ib)
    F_r, y_r, _, _ = task_dataset(task, rt.X, rt.labels)
    F_s, y_s, _, _ = task_dataset(task, sy.X, sy.labels)
    F_t, y_t, rows_t, _ = task_dataset(task, real_test.X, real_test.labels)
    keep = np.isfinite(y_t)
    F_t, y_t, rows_t = F_t[keep], y_t[keep].astype(int), rows_t[keep]
    s_trtr = fit_predict(F_r, y_r, F_t, model, seed)
    s_tstr = fit_predict(F_s, y_s, F_t, model, seed)
    g = _patient_groups(real_test, rows_t)
    trtr, tstr = auc_ci(y_t, s_trtr, g, n_boot, seed=seed), auc_ci(y_t, s_tstr, g, n_boot, seed=seed)
    delta = paired_delta_ci(y_t, s_trtr, s_tstr, g, n_boot, seed=seed)
    res = {"task": task.name, "kind": task.kind, "description": task.description, "source": task.source,
           "n_train_rows_real": int(np.isfinite(y_r).sum()), "n_train_rows_synth": int(np.isfinite(y_s).sum()),
           "prevalence_train_real": float(np.nanmean(y_r)), "prevalence_train_synth": float(np.nanmean(y_s)),
           "n_test_rows": int(len(y_t)), "prevalence_test": float(y_t.mean()),
           "auroc_trtr": trtr, "auroc_tstr": tstr,
           "auprc_trtr": float(average_precision_score(y_t, s_trtr)), "auprc_tstr": float(average_precision_score(y_t, s_tstr)),
           "delta_auroc": delta, "delong_p": delong_test(y_t, s_trtr, s_tstr)["p"],
           "relative_utility": float(tstr[0] / trtr[0]), "margin": margin, "non_inferior": non_inferior(delta[2], margin),
           "decision": ni_decision(delta, margin)}
    sub = {}
    for key in subgroup_keys:
        if key not in real_test.groups:
            continue
        gv = real_test.groups[key][rows_t]
        sub[key] = {}
        for val in sorted(set(gv)):
            sel = gv == val
            yy = y_t[sel]
            if sel.sum() < min_group or yy.sum() < min_pos or (len(yy) - yy.sum()) < min_pos:
                sub[key][val] = {"n": int(sel.sum()), "note": "too small"}
                continue
            d = paired_delta_ci(yy, s_trtr[sel], s_tstr[sel], None if g is None else g[sel], max(200, n_boot // 2), seed=seed)
            sub[key][val] = {"n": int(sel.sum()), "prevalence": float(yy.mean()),
                             "auroc_trtr": float(roc_auc_score(yy, s_trtr[sel])), "auroc_tstr": float(roc_auc_score(yy, s_tstr[sel])),
                             "delta_auroc": d}
    if sub:
        res["subgroups"] = sub
    return res


def tier1_forecast(task: Task, real_train: Cohort, real_test: Cohort, synth: Cohort, rel_margin: float = 0.05,
                   n_boot: int = 1000, seed: int = 0, size_match: bool = True) -> Dict:
    """Next-hour value forecasting (one-step predictive score): MAE on real test of a model trained on real
    (TRTR) or synthetic (TSTR) data, next to the last-value (persistence) baseline. Non-inferior if the
    upper CI bound of MAE(TSTR) - MAE(TRTR) is below rel_margin x MAE(TRTR)."""
    ia, ib = _size_match(len(real_train), len(synth), seed) if size_match else (np.arange(len(real_train)), np.arange(len(synth)))
    F_r, y_r, _, _ = task_dataset(task, real_train.X[ia])
    F_s, y_s, _, _ = task_dataset(task, synth.X[ib])
    F_t, y_t, rows_t, pers = task_dataset(task, real_test.X)
    kr, ks, kt = np.isfinite(y_r), np.isfinite(y_s), np.isfinite(y_t)
    reg = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, loss="absolute_error", random_state=seed)
    p_r = reg().fit(F_r[kr], y_r[kr]).predict(F_t[kt])
    p_s = reg().fit(F_s[ks], y_s[ks]).predict(F_t[kt])
    y, g = y_t[kt], _patient_groups(real_test, rows_t[kt])
    mae = lambda p: float(np.mean(np.abs(y - p)))
    delta = paired_mae_delta_ci(y, p_r, p_s, g, n_boot, seed=seed)
    margin = rel_margin * mae(p_r)
    return {"task": task.name, "kind": "forecast", "description": task.description, "source": task.source,
            "n_test_rows": int(kt.sum()), "mae_trtr": mae(p_r), "mae_tstr": mae(p_s), "mae_persistence": mae(pers[kt]),
            "delta_mae": delta, "margin": margin, "rel_margin": rel_margin, "non_inferior": non_inferior(delta[2], margin),
            "decision": ni_decision(delta, margin)}


def real_model_spread(task: Task, real_train: Cohort, real_test: Cohort, seeds: Sequence[int] = (0, 1, 2),
                      models: Sequence[str] = ("hgb", "lr")) -> Dict:
    """Real-data AUCs across seeds and model families (the anchor proposed in the ECG review)."""
    F_r, y_r, _, _ = task_dataset(task, real_train.X, real_train.labels)
    F_t, y_t, _, _ = task_dataset(task, real_test.X, real_test.labels)
    k = np.isfinite(y_t)
    aucs = {mdl: [float(roc_auc_score(y_t[k], fit_predict(F_r, y_r, F_t[k], mdl, s))) for s in (seeds if mdl == "hgb" else seeds[:1])]
            for mdl in models}
    seed_spread = max(a) - min(a) if (a := aucs.get("hgb")) else 0.0
    means = [np.mean(a) for a in aucs.values()]
    return {"aucs": aucs, "seed_spread": float(seed_spread), "family_spread": float(max(means) - min(means))}


def split_by_patient(cohort: Cohort, seed: int = 0) -> Tuple[Cohort, Cohort]:
    """Two disjoint halves of a cohort, split by patient."""
    rng = np.random.default_rng(seed)
    subs = np.unique(cohort.subjects if cohort.subjects is not None else np.arange(len(cohort)))
    rng.shuffle(subs)
    ids = cohort.subjects if cohort.subjects is not None else np.arange(len(cohort))
    inA = np.isin(ids, subs[: len(subs) // 2])
    return cohort.subset(np.flatnonzero(inA), cohort.name + "_A"), cohort.subset(np.flatnonzero(~inA), cohort.name + "_B")


def real_reference(real_train: Cohort, real_test: Cohort, tasks: Sequence[str] = DEFAULT_TASKS, n_boot: int = 1000,
                   seed: int = 0, rel_margin: float = 0.05, spread_seeds: Sequence[int] = (0, 1, 2),
                   verbose: bool = False) -> Dict[str, Dict]:
    """Real-vs-real reference for every task, independent of any generator (compute once, reuse for all):
    a model trained on one half of the real training patients plays 'TRTR', a model trained on the other
    half plays a perfect generator's 'TSTR'. Its delta CI shows what an ideal generator would score, so a
    task whose reference upper bound already reaches the margin cannot demonstrate non-inferiority
    (underpowered). Both halves are smaller than the full training set, so this is a conservative check."""
    A, B = split_by_patient(real_train, seed)
    out = {}
    for name in tasks:
        task = TASKS[name]
        if task.kind == "outcome" and task.outcome_key not in real_train.labels:
            continue
        if verbose:
            print("  reference:", name, flush=True)
        if task.kind == "forecast":
            r = tier1_forecast(task, A, real_test, B, rel_margin, n_boot, seed, size_match=False)
            out[name] = {"halves_delta": r["delta_mae"], "halves_mae": (r["mae_trtr"], r["mae_tstr"])}
        else:
            r = tier1_task(task, A, real_test, B, margin=1.0, n_boot=n_boot, seed=seed, size_match=False)
            out[name] = {"halves_delta": r["delta_auroc"], "halves_auroc": (r["auroc_trtr"][0], r["auroc_tstr"][0]),
                         **real_model_spread(task, real_train, real_test, spread_seeds)}
    return out


# =====================================================================================
# 6. Tier 2: fidelity on measured slots
# =====================================================================================
def measured(X: np.ndarray, sig: str) -> np.ndarray:
    v, m = split(X)
    return v[sig][m[sig]]


def mdd(real_vals: np.ndarray, syn_vals: np.ndarray, bins: int = 50) -> float:
    """Marginal distribution difference in the spirit of TSGBench (Ang et al., 2023): histogram of the
    synthetic values on bins taken from the real values (0.5th-99.5th percentile), mean absolute difference
    of the normalized histograms. Values outside the real range fall into the two edge bins."""
    lo, hi = np.quantile(real_vals, [0.005, 0.995])
    if hi <= lo:
        hi = lo + 1e-6
    edges = np.linspace(lo, hi, bins + 1)
    h = lambda x: np.histogram(np.clip(x, lo, hi), edges)[0] / max(len(x), 1)
    return float(np.mean(np.abs(h(syn_vals) - h(real_vals))))


def hourly_changes(X: np.ndarray, sig: str) -> np.ndarray:
    v, m = split(X)
    ok = m[sig][:, 1:] & m[sig][:, :-1]
    return (v[sig][:, 1:] - v[sig][:, :-1])[ok]


def acf_pooled(X: np.ndarray, sig: str, lags: Sequence[int] = (1, 2, 3, 6, 12, 24)) -> np.ndarray:
    """Lag-k autocorrelation over all pairs of measured hours k apart, after removing each stay's mean of
    measured values (pooled across stays, so between-stay differences do not inflate it)."""
    v, m = split(X)
    x, mk = v[sig], m[sig]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(np.where(mk, x, np.nan), axis=1, keepdims=True)
    z = np.where(mk, x - mu, 0.0)
    var = (z ** 2)[mk].mean() if mk.any() else np.nan
    out = []
    for k in lags:
        ok = mk[:, k:] & mk[:, :-k]
        out.append(float((z[:, k:] * z[:, :-k])[ok].mean() / var) if ok.any() and var > 0 else np.nan)
    return np.array(out)


def cross_correlations(X: np.ndarray, sigs: Sequence[str] = SIG) -> np.ndarray:
    """Pearson correlation between signals over hours where both are measured (pooled)."""
    v, m = split(X)
    C = np.eye(len(sigs))
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            ok = m[sigs[a]] & m[sigs[b]]
            if ok.sum() > 10:
                C[a, b] = C[b, a] = np.corrcoef(v[sigs[a]][ok], v[sigs[b]][ok])[0, 1]
            else:
                C[a, b] = C[b, a] = np.nan
    return C


def _w(a, b) -> float:
    return float(stats.wasserstein_distance(a, b)) if len(a) and len(b) else float("nan")


def missingness_profile(X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
    _, m = split(X)
    out = {}
    for s in SIG:
        mk = m[s]
        cnt = mk.sum(axis=1)
        first = np.where(mk.any(1), mk.argmax(1), -1)
        gaps = np.concatenate([np.diff(np.flatnonzero(r)) for r in mk]) if mk.any() else np.array([])
        out[s] = {"count": cnt, "first_hour": first[first >= 0], "gaps": gaps}
    return out


def validity(X: np.ndarray, channels: Dict) -> Dict[str, float]:
    """Physiological and preprocessing checks on measured slots."""
    v, m = split(X)
    both = m["sbp"] & m["dbp"]
    res = {"dbp_ge_sbp_share": float((v["dbp"] >= v["sbp"])[both].mean()) if both.any() else float("nan")}
    for s in SIG:
        x = v[s][m[s]]
        lo, hi = channels["clip_bounds"][s]
        fill = channels["leading_fill"][s]
        if x.size == 0:
            continue
        res[f"{s}_outside_bounds"] = float(((x < lo) | (x > hi)).mean())
        res[f"{s}_at_bounds"] = float(((np.isclose(x, lo)) | (np.isclose(x, hi))).mean())
        res[f"{s}_equals_fill_constant"] = float(np.isclose(x, fill, atol=1e-6).mean())
    return res


def tier2_signals(real: Cohort, other: Cohort) -> Dict[str, Dict]:
    """Per-signal fidelity of `other` against `real` (both on measured slots)."""
    out = {}
    lags = (1, 2, 3, 6, 12, 24)
    for s in SIG:
        a, b = measured(real.X, s), measured(other.X, s)
        if not len(b):
            out[s] = {"note": "no measured values"}
            continue
        iqr = float(np.subtract(*np.quantile(a, [0.75, 0.25]))) or 1.0
        qs = [0.01, 0.05, 0.5, 0.95, 0.99]
        da, db = hourly_changes(real.X, s), hourly_changes(other.X, s)
        out[s] = {"n_measured": int(len(b)), "measured_share": float(len(b) / (len(other) * N_T)),
                  "median": float(np.median(b)), "quantile_diff": dict(zip(map(str, qs), (np.quantile(b, qs) - np.quantile(a, qs)).round(4).tolist())),
                  "wasserstein": _w(a, b), "wasserstein_per_iqr": _w(a, b) / iqr,
                  "ks": float(stats.ks_2samp(a, b).statistic), "mdd": mdd(a, b),
                  "skew_diff": float(abs(stats.skew(b) - stats.skew(a))), "kurtosis_diff": float(abs(stats.kurtosis(b, fisher=False) - stats.kurtosis(a, fisher=False))),
                  "change_wasserstein": _w(da, db) if len(db) else float("nan"),
                  "acd": float(np.nanmean(np.abs(acf_pooled(other.X, s, lags) - acf_pooled(real.X, s, lags)))) if s in VITALS else float("nan")}
    return out


def tier2_missingness(real: Cohort, other: Cohort) -> Dict[str, Dict]:
    pa, pb = missingness_profile(real.X), missingness_profile(other.X)
    out = {}
    for s in SIG:
        out[s] = {"median_count_real": float(np.median(pa[s]["count"])), "median_count_other": float(np.median(pb[s]["count"])),
                  "count_wasserstein": _w(pa[s]["count"], pb[s]["count"]),
                  "first_hour_wasserstein": _w(pa[s]["first_hour"], pb[s]["first_hour"]),
                  "gap_wasserstein": _w(pa[s]["gaps"], pb[s]["gaps"])}
    return out


def discriminative_score(real: Cohort, other: Cohort, max_n: int = 10000, seed: int = 0) -> Dict[str, float]:
    """Post-hoc classifier real vs other (Yoon et al., 2019 used a 2-layer LSTM; here gradient boosting on
    per-stay summary features). Balanced classes, 70/30 split. Reported: held-out accuracy, |accuracy - 0.5|
    (0 = indistinguishable) and AUROC."""
    rng = np.random.default_rng(seed)
    n = min(len(real), len(other), max_n)
    A = summary_features(real.X[rng.choice(len(real), n, replace=False)])
    B = summary_features(other.X[rng.choice(len(other), n, replace=False)])
    F, y = np.vstack([A, B]), np.r_[np.zeros(n), np.ones(n)]
    Ftr, Fte, ytr, yte = train_test_split(F, y, test_size=0.3, stratify=y, random_state=seed)
    clf = make_classifier("hgb", seed).fit(Ftr, ytr)
    p = clf.predict_proba(Fte)[:, 1]
    acc = float(((p >= 0.5) == yte).mean())
    return {"accuracy": acc, "score_abs_acc_minus_half": abs(acc - 0.5), "auroc": float(roc_auc_score(yte, p)), "n_per_class": n}


class Embedding:
    """Fixed evaluation embedding: per-stay summary features (or forward-filled vital-sign trajectories),
    median-imputed and standardized with statistics of the REAL TRAINING data only.
    Alaa et al. (2022) learn a one-class embedding (Seq2Seq LSTM + one-class network); this simpler fixed
    embedding keeps the metrics reproducible across generators but is not their implementation. For the
    official version see Synthcity (Qian et al., 2023)."""

    def __init__(self, kind: str = "summary"):
        self.kind = kind

    def _raw(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "summary":
            return summary_features(X)
        v, m = split(X)
        cols = []
        for s in VITALS:                               # forward fill from measured values only
            x = np.where(m[s], v[s], np.nan)
            idx = np.where(m[s], np.arange(N_T), 0)
            np.maximum.accumulate(idx, axis=1, out=idx)
            ff = x[np.arange(len(x))[:, None], idx]
            ff[~np.maximum.accumulate(m[s], axis=1)] = np.nan
            cols.append(ff)
        return np.hstack(cols)

    def fit(self, X: np.ndarray) -> "Embedding":
        R = self._raw(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self.med = np.nan_to_num(np.nanmedian(R, axis=0))
            R = np.where(np.isfinite(R), R, self.med)
        self.mu, self.sd = R.mean(0), R.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        R = self._raw(X)
        R = np.where(np.isfinite(R), R, self.med)
        return (R - self.mu) / self.sd


def _nn_dist(ref: np.ndarray, query: np.ndarray, k: int = 1, exclude_self: bool = False) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + int(exclude_self)).fit(ref)
    d, _ = nn.kneighbors(query)
    return d[:, -1]


def alpha_beta_authenticity(real_emb: np.ndarray, syn_emb: np.ndarray, k: int = 5,
                            grid: Sequence[float] = tuple(np.linspace(0.05, 1.0, 20))) -> Dict:
    """Sample-level metrics of Alaa et al. (2022), on a fixed embedding with the hypersphere centred at the
    real mean. alpha-precision: share of synthetic samples inside the real alpha-quantile ball. beta-recall:
    share of real samples whose nearest synthetic sample inside the synthetic beta-quantile ball lies within
    the real sample's k-NN radius. IP = 1 - 2 x integral |P_alpha - alpha|, IR likewise. Authenticity: share of
    synthetic samples NOT closer to their nearest real training sample i* than i*'s own nearest real
    neighbour.
    Caveat (checked in test_mimic_utility.py): outside a learned low-dimensional embedding the beta-recall
    curve is not calibrated. With 8 or more dimensions, two samples from the SAME distribution give
    R_beta far above beta and an IR_beta well below 1, and IR_beta then cannot tell good coverage from
    mode collapse. Read these numbers against the real-vs-real reference, not against 1.0, and use
    recall_at_beta1 (share of real samples whose nearest synthetic sample lies within their k-NN radius),
    which falls with mode collapse."""
    c_r, c_g = real_emb.mean(0), syn_emb.mean(0)
    d_r = np.linalg.norm(real_emb - c_r, axis=1)
    d_g_to_r = np.linalg.norm(syn_emb - c_r, axis=1)
    d_g = np.linalg.norm(syn_emb - c_g, axis=1)
    nnd_k = _nn_dist(real_emb, real_emb, k=k, exclude_self=True)
    P, R = [], []
    for a in grid:
        P.append(float((d_g_to_r <= np.quantile(d_r, a)).mean()))
        inside = syn_emb[d_g <= np.quantile(d_g, a)]
        R.append(float((_nn_dist(inside, real_emb) <= nnd_k).mean()) if len(inside) else 0.0)
    P, R, g = np.array(P), np.array(R), np.array(grid)

    def integ(y):          # integral of |curve - alpha| over the grid, rescaled to the unit interval
        dev = np.abs(y - g)
        return float(np.sum((dev[1:] + dev[:-1]) / 2 * np.diff(g)) / (g[-1] - g[0]))
    nn = NearestNeighbors(n_neighbors=1).fit(real_emb)
    d_gj, i_star = nn.kneighbors(syn_emb)
    d_r_istar = _nn_dist(real_emb, real_emb, k=1, exclude_self=True)[i_star[:, 0]]
    return {"alpha_grid": g.round(3).tolist(), "alpha_precision_curve": P.round(4).tolist(), "beta_recall_curve": R.round(4).tolist(),
            "IP_alpha": 1 - 2 * integ(P), "IR_beta": 1 - 2 * integ(R), "precision_at_alpha1": float(P[-1]),
            "recall_at_beta1": float(R[-1]), "authenticity": float((d_gj[:, 0] > d_r_istar).mean()), "k": k}


def tier2(real_train: Cohort, real_test: Cohort, synth: Cohort, channels: Dict, max_n: int = 10000, seed: int = 0) -> Dict:
    """Every metric for synthetic vs real train, next to the reference real test vs real train."""
    rng = np.random.default_rng(seed)
    out = {"signals": {"synthetic": tier2_signals(real_train, synth), "reference": tier2_signals(real_train, real_test)},
           "missingness": {"synthetic": tier2_missingness(real_train, synth), "reference": tier2_missingness(real_train, real_test)},
           "validity": {"real_train": validity(real_train.X, channels), "synthetic": validity(synth.X, channels)}}
    Cr, Cs, Ct = cross_correlations(real_train.X), cross_correlations(synth.X), cross_correlations(real_test.X)
    iu = np.triu_indices(len(SIG), 1)
    out["cross_correlation"] = {"signals": SIG, "real_train": Cr.round(4).tolist(), "synthetic": Cs.round(4).tolist(),
                                "mean_abs_diff_synthetic": float(np.nanmean(np.abs(Cs - Cr)[iu])),
                                "mean_abs_diff_reference": float(np.nanmean(np.abs(Ct - Cr)[iu]))}
    out["discriminative"] = {"synthetic": discriminative_score(real_train, synth, max_n, seed),
                             "reference": discriminative_score(real_train, real_test, max_n, seed)}
    emb = Embedding("summary").fit(real_train.X)
    n = min(len(real_train), len(synth), len(real_test), max_n)
    R = emb.transform(real_train.X[rng.choice(len(real_train), n, replace=False)])
    out["sample_level"] = {"synthetic": alpha_beta_authenticity(R, emb.transform(synth.X[rng.choice(len(synth), n, replace=False)])),
                           "reference": alpha_beta_authenticity(R, emb.transform(real_test.X[rng.choice(len(real_test), n, replace=False)]))}
    return out


# =====================================================================================
# 7. Tier 3: clinical-event consistency
# =====================================================================================
def tier3(real_train: Cohort, real_test: Cohort, synth: Cohort, margin: float = 0.05, n_boot: int = 1000, seed: int = 0,
          outcome: str = "mortality") -> Dict:
    """Event rates over hours 0-72: synthetic vs real train (equivalence within +/- margin), with the real
    test vs real train gap as the sampling-noise reference. If the synthetic cohort carries the outcome
    label, also compare the outcome gradient (rate among deaths - rate among survivors)."""
    fr, fs, ft = event_flags(real_train.X), event_flags(synth.X), event_flags(real_test.X)
    res = {"margin": margin, "events": {}}
    for k in EVENTS:
        r = rate_gap(fr[k], fs[k], real_train.subjects, margin, n_boot, seed=seed)
        r["reference_gap"] = float(ft[k].mean() - fr[k].mean())
        res["events"][k] = r
    if outcome in synth.labels and outcome in real_train.labels:
        yr, ys = real_train.labels[outcome], synth.labels[outcome]
        grad = {}
        for k in EVENTS:
            gr = fr[k][yr == 1].mean() - fr[k][yr == 0].mean()
            gs = fs[k][ys == 1].mean() - fs[k][ys == 0].mean() if (ys == 1).any() and (ys == 0).any() else float("nan")
            grad[k] = {"gradient_real": float(gr), "gradient_synth": float(gs), "diff": float(gs - gr)}
        res["outcome_gradient"] = {"outcome": outcome, "rate_real": float(np.nanmean(yr)), "rate_synth": float(np.nanmean(ys)), "events": grad}
    return res


# =====================================================================================
# 8. Privacy (separate from utility)
# =====================================================================================
def _aa(T: np.ndarray, S: np.ndarray) -> float:
    """Nearest-neighbor adversarial accuracy AA_TS of Yale et al. (2019); 0.5 = indistinguishable."""
    d_ts = _nn_dist(S, T)
    d_tt = _nn_dist(T, T, exclude_self=True)
    d_st = _nn_dist(T, S)
    d_ss = _nn_dist(S, S, exclude_self=True)
    return float(0.5 * ((d_ts > d_tt).mean() + (d_st > d_ss).mean()))


def privacy(real_train: Cohort, real_test: Cohort, synth: Cohort, n: int = 5000, seed: int = 0,
            embeddings: Sequence[str] = ("summary", "raw_vitals")) -> Dict:
    """Distance-based privacy checks on a fixed embedding (fit on real train). Nearest-neighbour searches run
    against the FULL training set and the FULL synthetic set (an attacker sees the whole release); only the
    query points are sampled (n of each).
    - DCR: distance from synthetic stays to their closest real training stay, against the same distance for
      held-out real test stays (a fresh sample of patients). For a generator that does not memorize, about
      5% of synthetic stays fall below the hold-out 5th percentile.
    - Membership inference (full black-box, Chen et al., 2020): score each candidate by minus its distance to
      the nearest synthetic stay; AUROC for training members vs held-out non-members (0.5 = no signal).
      Accuracy at the median threshold is also given (as in PyHealth's calc_membership_inference).
    - NNAA (Yale et al., 2019) on balanced samples of n: AA with training stays, AA with test stays,
      privacy loss = AA_test - AA_train (about 0 = no memorization signal)."""
    rng = np.random.default_rng(seed)
    n = min(n, len(real_train), len(real_test), len(synth))
    i_mem = rng.choice(len(real_train), n, replace=False)
    i_non = rng.choice(len(real_test), n, replace=False)
    i_syn = rng.choice(len(synth), n, replace=False)
    out = {"n_queries": n, "n_train_reference": len(real_train), "n_synthetic_reference": len(synth)}
    for kind in embeddings:
        emb = Embedding("summary" if kind == "summary" else "raw").fit(real_train.X)
        Tr, Sy = emb.transform(real_train.X), emb.transform(synth.X)
        Te = emb.transform(real_test.X[i_non])
        dcr_s, dcr_t = _nn_dist(Tr, Sy[i_syn]), _nn_dist(Tr, Te)
        q5 = np.quantile(dcr_t, 0.05)
        d_mem, d_non = _nn_dist(Sy, Tr[i_mem]), _nn_dist(Sy, Te)
        y = np.r_[np.ones(n), np.zeros(n)]
        d_all = np.r_[d_mem, d_non]
        thr = np.median(d_all)
        aa_tr, aa_te = _aa(Tr[i_mem], Sy[i_syn]), _aa(Te, Sy[i_syn])
        out[kind] = {"dcr_median_synth": float(np.median(dcr_s)), "dcr_median_holdout": float(np.median(dcr_t)),
                     "share_synth_below_holdout_p5": float((dcr_s < q5).mean()), "exact_copies_share": float((dcr_s < 1e-8).mean()),
                     "mia_auroc": float(fast_auc(y, -d_all)), "mia_accuracy_median_threshold": float(((d_all <= thr) == (y == 1)).mean()),
                     "aa_train": aa_tr, "aa_test": aa_te, "privacy_loss": aa_te - aa_tr}
    return out


# =====================================================================================
# 9. Runner, report and command line
# =====================================================================================
def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not math.isfinite(float(o)) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    return o


def evaluate(real_train: Cohort, real_test: Cohort, synth: Cohort, channels: Optional[Dict] = None,
             tasks: Sequence[str] = DEFAULT_TASKS, margin: float = 0.02, rate_margin: float = 0.05,
             forecast_rel_margin: float = 0.05, n_boot: int = 1000, seed: int = 0,
             subgroup_keys: Sequence[str] = ("race_group", "insurance_group", "sex", "age_band"),
             run_tier2: bool = True, run_tier3: bool = True, run_privacy: bool = True, max_n: int = 10000,
             reference: Optional[Dict[str, Dict]] = None, match_resolution: bool = True, notes: Sequence[str] = (),
             verbose: bool = True) -> Dict:
    """Run the whole evaluation for one synthetic cohort.
    margin: pre-registered AUROC non-inferiority margin for every classification task (default 0.02).
    reference: output of real_reference() (computed once and shared across generators); when given, each
    task reports the perfect-generator delta and is flagged 'underpowered' if that reference cannot pass.
    match_resolution: snap synthetic values to the real value grid before labels and metrics (ValueGrid);
    validity checks are also reported on the raw generator output."""
    t0 = time.time()
    log = (lambda *a: print(f"[{time.time() - t0:6.0f}s]", *a, flush=True)) if verbose else (lambda *a: None)
    channels = channels or load_channels(None)
    rep = {"meta": {"n_real_train": len(real_train), "n_real_test": len(real_test), "n_synthetic": len(synth),
                    "synthetic_labels": sorted(synth.labels), "notes": list(notes), "seed": seed, "n_boot": n_boot,
                    "auroc_margin": margin, "forecast_rel_margin": forecast_rel_margin, "rate_margin": rate_margin,
                    "resolution_matching": bool(match_resolution)},
           "tier1": [], "skipped": []}
    raw_validity = validity(synth.X, channels)
    if match_resolution:
        Xs, shift = ValueGrid(real_train.X).snap(synth.X)
        synth = Cohort(Xs, synth.subjects, synth.labels, synth.groups, synth.name)
        rep["meta"]["resolution_shift_mean_abs"] = shift
    for name in tasks:
        task = TASKS[name]
        if task.kind == "outcome" and task.outcome_key not in synth.labels:
            rep["skipped"].append({"task": name, "reason": "synthetic data carry no label " + str(task.outcome_key)})
            continue
        log("tier 1:", name)
        if task.kind == "forecast":
            r = tier1_forecast(task, real_train, real_test, synth, forecast_rel_margin, n_boot, seed)
        else:
            r = tier1_task(task, real_train, real_test, synth, margin, n_boot, seed, subgroup_keys=subgroup_keys)
        if reference and name in reference:
            ref = reference[name]
            r["reference"] = ref
            r["underpowered"] = bool(ref["halves_delta"][2] >= r["margin"])
        rep["tier1"].append(r)
    if run_tier2:
        log("tier 2")
        rep["tier2"] = tier2(real_train, real_test, synth, channels, max_n, seed)
        rep["tier2"]["validity"]["synthetic_raw_output"] = raw_validity
    if run_tier3:
        log("tier 3")
        rep["tier3"] = tier3(real_train, real_test, synth, rate_margin, n_boot, seed)
    if run_privacy:
        log("privacy")
        rep["privacy"] = privacy(real_train, real_test, synth, min(5000, max_n), seed)
    rep["meta"]["runtime_s"] = round(time.time() - t0, 1)
    return _jsonable(rep)


def evaluate_many(real_train: Cohort, real_test: Cohort, synthetic: Dict[str, Cohort], channels: Optional[Dict] = None,
                  tasks: Sequence[str] = DEFAULT_TASKS, compute_reference: bool = True, **kw) -> Dict[str, Dict]:
    """Evaluate several generators (e.g. SSSD-ECG, TTS-GAN, TransFusion, Diffusion-TS) with one shared
    real-vs-real reference, so every generator is judged against the same yardstick."""
    ref = real_reference(real_train, real_test, tasks, kw.get("n_boot", 1000), kw.get("seed", 0),
                         kw.get("forecast_rel_margin", 0.05), verbose=kw.get("verbose", True)) if compute_reference else None
    return {name: evaluate(real_train, real_test, syn, channels, tasks, reference=ref, **kw) for name, syn in synthetic.items()}


def _f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def _ref(r: Dict, d: int = 3) -> str:
    if "reference" not in r:
        return "-"
    h = r["reference"]["halves_delta"]
    return f"{_f(h[0], d)} ({_f(h[1], d)} to {_f(h[2], d)})" + (" UNDERPOWERED" if r.get("underpowered") else "")


def report_markdown(rep: Dict) -> str:
    L = ["# Synthetic MIMIC-IV utility report", "",
         f"Real train {rep['meta']['n_real_train']} stays, real test {rep['meta']['n_real_test']}, synthetic {rep['meta']['n_synthetic']}. "
         f"AUROC margin {rep['meta']['auroc_margin']}, forecast margin {rep['meta']['forecast_rel_margin']:.0%} of TRTR MAE, "
         f"event-rate margin +/- {rep['meta']['rate_margin']}. Runtime {rep['meta'].get('runtime_s')} s.",
         ("Synthetic values were snapped to the real value grid before labels and metrics (resolution matching). "
          if rep["meta"].get("resolution_matching") else "No resolution matching. ") +
         "Decision: non-inferior = upper CI bound of (TRTR - TSTR) below the margin; inferior = lower bound above it; "
         "otherwise inconclusive. 'Perfect-generator delta' trains on held-out real patients instead of synthetic data; "
         "UNDERPOWERED marks tasks where even that cannot pass."]
    for n_ in rep["meta"].get("notes", []):
        L.append(f"- Note: {n_}")
    L += ["", "## Tier 1: task utility (TSTR vs TRTR, real test set)", "",
          "| Task | Prevalence real / synth | AUROC TRTR | AUROC TSTR | Delta (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |", "|---|---|---|---|---|---|---|---|"]
    for r in rep["tier1"]:
        if r["kind"] == "forecast":
            continue
        d = r["delta_auroc"]
        L.append(f"| {r['task']} | {_f(r['prevalence_train_real'])} / {_f(r['prevalence_train_synth'])} | {_f(r['auroc_trtr'][0])} | "
                 f"{_f(r['auroc_tstr'][0])} | {_f(d[0])} ({_f(d[1])} to {_f(d[2])}) | {_f(r['margin'])} | {r['decision']} | {_ref(r)} |")
    fc = [r for r in rep["tier1"] if r["kind"] == "forecast"]
    if fc:
        L += ["", "| Forecast task | MAE TRTR | MAE TSTR | MAE last value | Delta MAE (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |", "|---|---|---|---|---|---|---|---|"]
        for r in fc:
            d = r["delta_mae"]
            L.append(f"| {r['task']} | {_f(r['mae_trtr'], 2)} | {_f(r['mae_tstr'], 2)} | {_f(r['mae_persistence'], 2)} | "
                     f"{_f(d[0], 2)} ({_f(d[1], 2)} to {_f(d[2], 2)}) | {_f(r['margin'], 2)} | {r['decision']} | {_ref(r, 2)} |")
    sub_lines = []
    for r in rep["tier1"]:
        cells = [(k, v, g) for k, grp in r.get("subgroups", {}).items() for v, g in grp.items() if "delta_auroc" in g]
        if cells:
            k, v, g = max(cells, key=lambda c: c[2]["delta_auroc"][0])
            d = g["delta_auroc"]
            sub_lines.append(f"| {r['task']} | {k} = {v} | {g['n']} | {_f(g['auroc_trtr'])} | {_f(g['auroc_tstr'])} | "
                             f"{_f(d[0])} ({_f(d[1])} to {_f(d[2])}) |")
    if sub_lines:
        L += ["", "Subgroup with the largest TRTR - TSTR delta for each task (all subgroups are in the JSON):", "",
              "| Task | Subgroup | Test stays | AUROC TRTR | AUROC TSTR | Delta (95% CI) |", "|---|---|---|---|---|---|"] + sub_lines
    for s in rep.get("skipped", []):
        L.append(f"- Skipped {s['task']}: {s['reason']}")
    if "tier2" in rep:
        t2 = rep["tier2"]
        L += ["", "## Tier 2: fidelity on measured slots (synthetic vs real train; reference = real test vs real train)", "",
              "| Signal | Wasserstein / IQR | ref | MDD | ref | Hourly change W | ref | ACD | ref | Median count synth / real |", "|---|---|---|---|---|---|---|---|---|---|"]
        for s in SIG:
            a, b = t2["signals"]["synthetic"].get(s, {}), t2["signals"]["reference"].get(s, {})
            mm = t2["missingness"]["synthetic"][s]
            if "note" in a:
                L.append(f"| {s} | {a['note']} |")
                continue
            L.append(f"| {s} | {_f(a['wasserstein_per_iqr'])} | {_f(b['wasserstein_per_iqr'])} | {_f(a['mdd'], 4)} | {_f(b['mdd'], 4)} | "
                     f"{_f(a['change_wasserstein'], 2)} | {_f(b['change_wasserstein'], 2)} | {_f(a['acd'])} | {_f(b['acd'])} | "
                     f"{_f(mm['median_count_other'], 0)} / {_f(mm['median_count_real'], 0)} |")
        cc, ds = t2["cross_correlation"], t2["discriminative"]
        sl = t2["sample_level"]
        L += ["", f"- Cross-signal correlation, mean absolute difference: synthetic {_f(cc['mean_abs_diff_synthetic'])}, reference {_f(cc['mean_abs_diff_reference'])}",
              f"- Discriminative score |accuracy - 0.5|: synthetic {_f(ds['synthetic']['score_abs_acc_minus_half'])} (AUROC {_f(ds['synthetic']['auroc'])}), "
              f"reference {_f(ds['reference']['score_abs_acc_minus_half'])} (AUROC {_f(ds['reference']['auroc'])})",
              f"- IP-alpha / IR-beta / recall at beta = 1 / authenticity: synthetic {_f(sl['synthetic']['IP_alpha'])} / {_f(sl['synthetic']['IR_beta'])} / "
              f"{_f(sl['synthetic']['recall_at_beta1'])} / {_f(sl['synthetic']['authenticity'])}; reference {_f(sl['reference']['IP_alpha'])} / "
              f"{_f(sl['reference']['IR_beta'])} / {_f(sl['reference']['recall_at_beta1'])} / {_f(sl['reference']['authenticity'])} (read against the reference, not 1.0)",
              f"- DBP >= SBP share: synthetic {_f(t2['validity']['synthetic']['dbp_ge_sbp_share'], 4)}, real {_f(t2['validity']['real_train']['dbp_ge_sbp_share'], 4)}"]
        raw = t2["validity"].get("synthetic_raw_output")
        if raw:
            out_b = {k[:-len('_outside_bounds')]: v for k, v in raw.items() if k.endswith("_outside_bounds") and v and v > 0}
            L.append("- Raw generator output outside the clip bounds: " + (", ".join(f"{k} {v:.4f}" for k, v in out_b.items()) if out_b else "none"))
    if "tier3" in rep:
        t3 = rep["tier3"]
        L += ["", f"## Tier 3: clinical-event rates, hours 0-72 (equivalence margin +/- {t3['margin']})", "",
              "| Event | Real train | Synthetic | Gap (95% CI) | Reference gap | Equivalent |", "|---|---|---|---|---|---|"]
        for k, r in t3["events"].items():
            L.append(f"| {k} | {_f(r['rate_real'])} | {_f(r['rate_synth'])} | {_f(r['gap'])} ({_f(r['ci'][0])} to {_f(r['ci'][1])}) | "
                     f"{_f(r['reference_gap'])} | {'yes' if r['equivalent'] else 'NO'} |")
        if "outcome_gradient" in t3:
            og = t3["outcome_gradient"]
            L += ["", f"Outcome gradient for {og['outcome']} (event rate among deaths minus survivors): "
                  + "; ".join(f"{k} real {_f(v['gradient_real'])} vs synth {_f(v['gradient_synth'])}" for k, v in og["events"].items())]
    if "privacy" in rep:
        L += ["", "## Privacy (not part of utility)", "",
              "| Embedding | DCR median synth / holdout | Synth below holdout 5th pct | Exact-copy share | MIA AUROC | NNAA privacy loss |", "|---|---|---|---|---|---|"]
        for k, r in rep["privacy"].items():
            if not isinstance(r, dict):
                continue
            L.append(f"| {k} | {_f(r['dcr_median_synth'])} / {_f(r['dcr_median_holdout'])} | {_f(r['share_synth_below_holdout_p5'])} | "
                     f"{_f(r['exact_copies_share'])} | {_f(r['mia_auroc'])} | {_f(r['privacy_loss'])} |")
        L.append("\nExpected for a generator that does not memorize: share below the 5th percentile near 0.05, MIA AUROC near 0.5, privacy loss near 0.")
    return "\n".join(L)


def main(argv: Optional[Sequence[str]] = None):
    p = argparse.ArgumentParser(description="Utility evaluation for synthetic MIMIC-IV tensors (stays x 16 x 72).")
    p.add_argument("--train", required=True); p.add_argument("--test", required=True)
    p.add_argument("--stays-train", required=True); p.add_argument("--stays-test", required=True)
    p.add_argument("--subjects-train", required=True); p.add_argument("--subjects-test", required=True)
    p.add_argument("--annotations", required=True); p.add_argument("--channels")
    p.add_argument("--synthetic", required=True, help=".npy (or .npz with key X) in original units")
    p.add_argument("--synthetic-labels", help=".npz with labels such as mortality (conditional generators)")
    p.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    p.add_argument("--margin", type=float, default=0.02, help="pre-registered AUROC non-inferiority margin")
    p.add_argument("--reference", help="JSON from a previous run's real reference (skips recomputing it)")
    p.add_argument("--no-reference", action="store_true", help="skip the real-vs-real reference")
    p.add_argument("--rate-margin", type=float, default=0.05)
    p.add_argument("--n-boot", type=int, default=1000); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-privacy", action="store_true"); p.add_argument("--no-tier2", action="store_true")
    p.add_argument("--no-resolution-matching", action="store_true", help="evaluate raw continuous values")
    p.add_argument("--out", default="mimic_utility_report")
    a = p.parse_args(argv)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    channels = load_channels(a.channels)
    rtr = load_real(a.train, a.stays_train, a.subjects_train, a.annotations, "real_train")
    rte = load_real(a.test, a.stays_test, a.subjects_test, a.annotations, "real_test")
    syn, notes = load_synthetic(a.synthetic, a.synthetic_labels)
    ref = None
    if a.reference:
        with open(a.reference) as f:
            ref = json.load(f)
    elif not a.no_reference:
        ref = _jsonable(real_reference(rtr, rte, a.tasks, a.n_boot, a.seed, verbose=True))
        with open(a.out + "_reference.json", "w") as f:
            json.dump(ref, f, indent=1)
    rep = evaluate(rtr, rte, syn, channels, a.tasks, a.margin, rate_margin=a.rate_margin, n_boot=a.n_boot, seed=a.seed,
                   run_tier2=not a.no_tier2, run_privacy=not a.no_privacy, reference=ref,
                   match_resolution=not a.no_resolution_matching, notes=notes)
    with open(a.out + ".json", "w") as f:
        json.dump(rep, f, indent=1)
    with open(a.out + ".md", "w") as f:
        f.write(report_markdown(rep))
    print(f"wrote {a.out}.json and {a.out}.md")


if __name__ == "__main__":
    main()
