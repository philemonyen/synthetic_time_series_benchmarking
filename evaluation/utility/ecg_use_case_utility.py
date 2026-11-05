"""
Reference implementation for the four PTB-XL use cases that can be tested directly:

    1. Acute coronary syndromes (MI and ischemic ST-T changes)
    2. Arrhythmia and conduction disease
    3. Routine baseline and triage (NORM vs abnormal; the six abnormalities of Ribeiro et al., 2020)
    4. PTB-XL research benchmark tasks (Strodthoff et al., 2021)

Sections
    1. PTB-XL labels and task definitions
    2. Feature extraction (cleaning, R peaks, median beats, multi-lead wave boundaries, measurements)
    3. Rule-based clinical criteria (thresholds are configurable; sources in comments)
    4. Utility metrics (task-based: TRTR / TSTR; feature fidelity; criterion rates; DeLong; bootstrap)

Status: a standalone reference, not a validated measurement program.
  - Feature extraction was checked on ONE real 12-lead ECG (PTB Diagnostic ECG Database record s0010_re from
    the wfdb-python sample data; not PTB-XL): three 10-s windows at 500 Hz and 100 Hz gave consistent
    intervals and criteria. That shows repeatability, not accuracy.
  - It has NOT been compared with the PTB-XL+ reference measurements (12SL, Uni-G, ECGDeli). Do that first,
    calibrate the heuristic parameters (qrs_frac, t_end_quantile) on the training folds, then freeze them.
  - Labels, criteria logic and statistics are covered by test_module.py.
Usage: run the SAME code with the SAME parameters on real and synthetic ECGs (same sampling rate, mV units).

Dependencies: numpy, pandas, scipy, scikit-learn, neurokit2 (tested with 0.2.13), wfdb (only for loading PTB-XL).
"""
from __future__ import annotations

import ast
import warnings
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal as sps
from scipy import stats
from sklearn.metrics import roc_auc_score

# =====================================================================================
# 1. PTB-XL labels and task definitions
# =====================================================================================
# Mapping copied from PTB-XL's scp_statements.csv (PhysioNet v1.0.3):
# diagnostic statement -> (diagnostic_class = superclass, diagnostic_subclass)
DIAGNOSTIC: Dict[str, Tuple[str, str]] = {
    "NDT": ("STTC", "STTC"), "NST_": ("STTC", "NST_"), "DIG": ("STTC", "STTC"), "LNGQT": ("STTC", "STTC"),
    "NORM": ("NORM", "NORM"), "IMI": ("MI", "IMI"), "ASMI": ("MI", "AMI"), "LVH": ("HYP", "LVH"),
    "LAFB": ("CD", "LAFB/LPFB"), "ISC_": ("STTC", "ISC_"), "IRBBB": ("CD", "IRBBB"), "1AVB": ("CD", "_AVB"),
    "IVCD": ("CD", "IVCD"), "ISCAL": ("STTC", "ISCA"), "CRBBB": ("CD", "CRBBB"), "CLBBB": ("CD", "CLBBB"),
    "ILMI": ("MI", "IMI"), "LAO/LAE": ("HYP", "LAO/LAE"), "AMI": ("MI", "AMI"), "ALMI": ("MI", "AMI"),
    "ISCIN": ("STTC", "ISCI"), "INJAS": ("MI", "AMI"), "LMI": ("MI", "LMI"), "ISCIL": ("STTC", "ISCI"),
    "LPFB": ("CD", "LAFB/LPFB"), "ISCAS": ("STTC", "ISCA"), "INJAL": ("MI", "AMI"), "ISCLA": ("STTC", "ISCA"),
    "RVH": ("HYP", "RVH"), "ANEUR": ("STTC", "STTC"), "RAO/RAE": ("HYP", "RAO/RAE"), "EL": ("STTC", "STTC"),
    "WPW": ("CD", "WPW"), "ILBBB": ("CD", "ILBBB"), "IPLMI": ("MI", "IMI"), "ISCAN": ("STTC", "ISCA"),
    "IPMI": ("MI", "IMI"), "SEHYP": ("HYP", "SEHYP"), "INJIN": ("MI", "IMI"), "INJLA": ("MI", "AMI"),
    "PMI": ("MI", "PMI"), "3AVB": ("CD", "_AVB"), "INJIL": ("MI", "IMI"), "2AVB": ("CD", "_AVB"),
}
FORM: List[str] = ["NDT", "NST_", "DIG", "LNGQT", "ABQRS", "PVC", "STD_", "VCLVH", "QWAVE", "LOWT", "NT_",
                   "PAC", "LPR", "INVT", "LVOLT", "HVOLT", "TAB_", "STE_", "PRC(S)"]
RHYTHM: List[str] = ["SR", "AFIB", "STACH", "SARRH", "SBRAD", "PACE", "SVARR", "BIGU", "AFLT", "SVTAC",
                     "PSVT", "TRIGU"]
SUPERCLASSES: List[str] = ["NORM", "MI", "STTC", "CD", "HYP"]
ALL_STATEMENTS: List[str] = sorted(set(DIAGNOSTIC) | set(FORM) | set(RHYTHM))
assert len(DIAGNOSTIC) == 44 and len(FORM) == 19 and len(RHYTHM) == 12 and len(ALL_STATEMENTS) == 71


@dataclass(frozen=True)
class Task:
    """A utility test task. `requires` says which labels the synthetic ECGs must carry."""
    name: str
    use_case: str
    level: str                      # 'superclass' | 'subclass' | 'statement'
    classes: Tuple[str, ...]
    requires: str                   # 'superclass' or 'statement'
    note: str = ""


TASKS: Dict[str, Task] = {t.name: t for t in [
    # Use case 1: acute coronary syndromes
    Task("acs_superclass", "ACS", "superclass", ("MI", "STTC"), "superclass"),
    Task("acs_mi_location", "ACS", "subclass", ("AMI", "IMI", "LMI", "PMI"), "statement"),
    Task("acs_form", "ACS", "statement", ("STE_", "STD_", "QWAVE", "INVT"), "statement"),
    # Use case 2: arrhythmia and conduction disease
    Task("conduction_superclass", "Arrhythmia/conduction", "superclass", ("CD",), "superclass"),
    Task("conduction_statements", "Arrhythmia/conduction", "statement",
         ("1AVB", "2AVB", "3AVB", "LAFB", "LPFB", "IRBBB", "CRBBB", "ILBBB", "CLBBB", "IVCD", "WPW"), "statement"),
    Task("rhythm", "Arrhythmia/conduction", "statement", tuple(RHYTHM), "statement",
         "Rhythm statements belong to no superclass."),
    # Use case 3: routine baseline and triage
    Task("triage_norm", "Triage", "superclass", ("ABNORMAL",), "superclass",
         "Binary: 1 = any of MI/STTC/CD/HYP, 0 = NORM only. See build_triage_labels()."),
    Task("triage_ribeiro6", "Triage", "statement", ("1AVB", "CRBBB", "CLBBB", "SBRAD", "AFIB", "STACH"),
         "statement"),
    # Use case 4: PTB-XL benchmark tasks (Strodthoff et al., 2021)
    Task("bench_all", "Benchmark", "statement", tuple(ALL_STATEMENTS), "statement"),
    Task("bench_diagnostic", "Benchmark", "statement", tuple(sorted(DIAGNOSTIC)), "statement"),
    Task("bench_subdiagnostic", "Benchmark", "subclass", tuple(sorted({s for _, s in DIAGNOSTIC.values()})),
         "statement"),
    Task("bench_superdiagnostic", "Benchmark", "superclass", tuple(SUPERCLASSES), "superclass"),
    Task("bench_form", "Benchmark", "statement", tuple(FORM), "statement"),
    Task("bench_rhythm", "Benchmark", "statement", tuple(RHYTHM), "statement"),
]}


def parse_scp_codes(value) -> Dict[str, float]:
    """PTB-XL stores scp_codes as a string like "{'NORM': 100.0, 'SR': 0.0}" (likelihood 0 = unknown)."""
    return value if isinstance(value, dict) else ast.literal_eval(value)


def codes_to_labels(scp: Dict[str, float], level: str, min_likelihood: float = 0.0) -> set:
    """Map one record's statements to labels at 'statement', 'subclass' or 'superclass' level.
    min_likelihood=0 keeps every listed statement (PTB-XL sets likelihood 0 when it is unknown)."""
    codes = [c for c, lik in scp.items() if lik >= min_likelihood]
    if level == "statement":
        return {c for c in codes if c in ALL_STATEMENTS}
    if level == "subclass":
        return {DIAGNOSTIC[c][1] for c in codes if c in DIAGNOSTIC}
    if level == "superclass":
        return {DIAGNOSTIC[c][0] for c in codes if c in DIAGNOSTIC}
    raise ValueError(level)


def build_labels(scp_series: Iterable, task: Task, min_likelihood: float = 0.0) -> np.ndarray:
    """Multi-hot label matrix (n_records x n_classes) for a task."""
    rows = []
    for v in scp_series:
        labs = codes_to_labels(parse_scp_codes(v), task.level, min_likelihood)
        rows.append([1 if c in labs else 0 for c in task.classes])
    return np.asarray(rows, dtype=int)


def build_triage_labels(scp_series: Iterable, conflict: str = "abnormal",
                        abnormal_rhythms: Optional[Sequence[str]] = None) -> np.ndarray:
    """NORM vs abnormal. Returns 1 (abnormal), 0 (normal) or -1 (exclude).
    Default (superclass level): abnormal = any of MI/STTC/CD/HYP; normal = NORM only; records without any
    diagnostic superclass are excluded. conflict='abnormal' counts NORM + another superclass as abnormal,
    conflict='exclude' drops them. Rhythm statements belong to no superclass, so a record with NORM and
    AFIB is 'normal' here; pass abnormal_rhythms (e.g. ("AFIB", "AFLT")) to count such rhythms as abnormal.
    Which rhythms count (e.g. STACH, SBRAD, SARRH) must be decided in advance."""
    out = []
    for v in scp_series:
        scp = parse_scp_codes(v)
        sup = codes_to_labels(scp, "superclass")
        abnormal = bool(sup & {"MI", "STTC", "CD", "HYP"})
        if abnormal_rhythms and set(scp) & set(abnormal_rhythms):
            out.append(1)
        elif not sup:
            out.append(-1)
        elif "NORM" in sup and abnormal:
            out.append(1 if conflict == "abnormal" else -1)
        else:
            out.append(1 if abnormal else 0)
    return np.asarray(out, dtype=int)


def recommended_split(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index, pd.Index]:
    """PTB-XL recommendation: folds 1-8 train, 9 validation, 10 test (patient-wise folds).
    Train the GENERATOR on folds 1-8 only, so fold 10 stays unseen in TSTR."""
    return (df.index[df.strat_fold <= 8], df.index[df.strat_fold == 9], df.index[df.strat_fold == 10])


def load_ptbxl(root: str, sampling_rate: int = 100) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load PTB-XL metadata and signals with wfdb. Returns (df, signals[n, T, 12], lead_names)."""
    import wfdb  # imported lazily so the rest of the module works without wfdb
    df = pd.read_csv(f"{root}/ptbxl_database.csv", index_col="ecg_id")
    df["scp_codes"] = df["scp_codes"].apply(parse_scp_codes)
    col = "filename_lr" if sampling_rate == 100 else "filename_hr"
    sigs, names = [], None
    for f in df[col]:
        x, meta = wfdb.rdsamp(f"{root}/{f}")
        sigs.append(x)
        names = names or meta["sig_name"]
    return df, np.stack(sigs), list(names)


# =====================================================================================
# 2. Feature extraction (NeuroKit2)
# =====================================================================================
LEADS_12 = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def normalize_lead(name: str) -> str:
    """Map header spellings such as 'AVR' to 'aVR'. Always select leads by name, not by position."""
    up = name.strip().upper()
    return {"AVR": "aVR", "AVL": "aVL", "AVF": "aVF"}.get(up, up)


def _nanmedian(x) -> float:
    x = np.asarray(x, dtype=float)
    return float(np.nanmedian(x)) if np.isfinite(x).any() else float("nan")


INDEPENDENT_LEADS = ("I", "II", "V1", "V2", "V3", "V4", "V5", "V6")   # the other 4 limb leads are derived


def clean_ecg(x: np.ndarray, fs: int, powerline: float = 50.0) -> np.ndarray:
    """Baseline removal with a zero-phase 0.5 Hz Butterworth high-pass (order 5, as in NeuroKit2's default
    cleaning) plus a zero-phase 50 Hz notch when the sampling rate allows it. NeuroKit2's default
    'powerline' step is a moving average over one mains period, which also smooths the QRS (20 ms at
    500 Hz), so it is replaced here by a notch. At 100 Hz, 50 Hz is the Nyquist frequency and no notch is
    applied. Use the SAME cleaning for real and synthetic ECGs."""
    import neurokit2 as nk
    y = nk.signal_filter(np.asarray(x, dtype=float), sampling_rate=fs, lowcut=0.5, method="butterworth", order=5)
    if fs >= 2 * powerline + 20:
        b, a = sps.iirnotch(w0=powerline, Q=30.0, fs=fs)
        y = sps.filtfilt(b, a, y)
    return y


def _multilead_p_onsets(clean: Dict[str, np.ndarray], r: np.ndarray, fs: int, leads: Sequence[str],
                        min_leads: int = 3) -> Tuple[np.ndarray, Dict[str, float]]:
    """P-wave onset per beat. Each lead is delineated with NeuroKit2's wavelet method ('dwt') on shared
    R peaks; the onsets are combined with the cross-lead median when >= `min_leads` leads found one
    (NaN otherwise). Single-lead delineation is error-prone, the cross-lead median is more stable."""
    import neurokit2 as nk
    rows = []
    for name in leads:
        try:
            _, wav = nk.ecg_delineate(clean[name], r, sampling_rate=fs, method="dwt")
        except Exception:
            continue
        v = np.asarray(wav.get("ECG_P_Onsets", []), dtype=float)
        rows.append(v if len(v) == len(r) else np.full(len(r), np.nan))
    a = np.vstack(rows) if rows else np.full((1, len(r)), np.nan)
    enough = np.isfinite(a).sum(axis=0) >= min_leads
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        p_on = np.where(enough, np.nanmedian(a, axis=0), np.nan)
        spread = np.nanmedian(np.abs(a - np.nanmedian(a, axis=0)), axis=0) / fs * 1000
    return p_on, {"p_onset_found": float(np.mean(enough)), "p_onset_lead_spread_ms": _nanmedian(spread)}


def _qrs_bounds_spatial_velocity(tm: Dict[str, np.ndarray], fs: int, t0: int,
                                 frac: float = 0.10) -> Tuple[int, int]:
    """Global QRS onset and offset (sample indices) on the median beats.
    Spatial velocity = Euclidean norm of the time derivatives of the independent leads (I, II, V1-V6),
    smoothed over 10 ms. Threshold = base + frac * (peak - base), with base = median velocity in the
    first 50 ms of the window. The QRS is the supra-threshold run around the velocity peak (gaps
    < 20 ms are bridged). A heuristic: check it against reference measurements before relying on it."""
    use = [l for l in INDEPENDENT_LEADS if l in tm] or list(tm)
    a = np.vstack([tm[l] for l in use])
    sv = np.sqrt((np.gradient(a, axis=1) ** 2).sum(axis=0)) * fs          # mV/s
    k = max(1, int(round(0.010 * fs)))
    sv = np.convolve(sv, np.ones(k) / k, mode="same")
    lo, hi = max(0, t0 - int(0.16 * fs)), min(len(sv) - 1, t0 + int(0.20 * fs))
    base = float(np.median(sv[: max(3, int(0.05 * fs))]))
    thr = base + frac * (float(sv[lo: hi + 1].max()) - base)
    idx = np.where(sv[lo: hi + 1] >= thr)[0] + lo
    gap = max(1, int(round(0.020 * fs)))
    br = np.where(np.diff(idx) > gap)[0]
    starts, ends = np.r_[idx[0], idx[br + 1]], np.r_[idx[br], idx[-1]]
    pk = lo + int(np.argmax(sv[lo: hi + 1]))
    j = int(np.searchsorted(starts, pk, side="right") - 1)
    return int(starts[j]), int(ends[j])


def _tangent_t_end(beat: np.ndarray, base: float, fs: int, j_i: int, rr_s: float,
                   min_amp_mV: float = 0.05) -> Optional[Tuple[float, float]]:
    """T-wave end on one lead's median beat by the tangent method: the steepest slope of the T wave's
    terminal limb is extended to the isoelectric level (here the QRS-onset level). Handles upright and
    inverted T waves. Returns (t_end index, signed T amplitude in mV) or None for a flat T wave."""
    lo = j_i + int(0.06 * fs)
    hi = min(len(beat) - 1, j_i + int(min(0.40, 0.6 * rr_s) * fs))
    if hi - lo < 3:
        return None
    apex = lo + int(np.argmax(np.abs(beat[lo: hi + 1] - base)))
    amp = float(beat[apex] - base)
    if abs(amp) < min_amp_mV:
        return None
    d = np.gradient(beat)                                   # mV per sample
    stop = min(len(beat) - 1, apex + int(0.20 * fs))
    if stop - apex < 2:
        return None
    s = apex + int(np.argmax(-np.sign(amp) * d[apex: stop + 1]))
    if d[s] == 0 or np.sign(d[s]) == np.sign(amp):
        return None
    return float(s + (base - beat[s]) / d[s]), amp


def extract_features(signal: np.ndarray, lead_names: Sequence[str], fs: int, ref_lead: str = "II",
                     pre_s: float = 0.40, post_s: float = 0.60, st_offset_ms: float = 60.0,
                     qrs_frac: float = 0.10, t_end_quantile: float = 50.0,
                     delineation_lowpass_hz: float = 40.0, q_min_mV: float = 0.02) -> Dict:
    """Clinical features from one 12-lead ECG (signal: samples x leads, in mV).

    Pipeline: cleaning (clean_ecg) -> R peaks on a reference lead (NeuroKit2) -> median beat per lead -> global
    measurements on the median beats. QRS onset/offset come from the multi-lead spatial velocity, the
    T-wave end from the tangent method per lead (combined across leads with `t_end_quantile`), and the
    P onset from NeuroKit2 wavelet delineation (cross-lead median). ST deviation is measured at the
    J point (QRS offset) relative to the QRS-onset level, the reference recommended by the Fourth
    Universal Definition of MI; a J+60 ms value is also returned.
    Run the SAME function on real and synthetic ECGs, and first check it against reference
    measurements (e.g., the 12SL / Uni-G / ECGDeli features in PTB-XL+) on real records.
    """
    import neurokit2 as nk
    x = np.asarray(signal, dtype=float)
    leads = [normalize_lead(n) for n in lead_names]
    clean = {name: clean_ecg(x[:, i], fs) for i, name in enumerate(leads)}
    # Slopes are inflated by noise, so R peaks, wave boundaries and the low-frequency P and T waves use a
    # 40 Hz low-passed copy; QRS and ST amplitudes are read from the cleaned signal without low-pass.
    hc = min(delineation_lowpass_hz, 0.45 * fs)
    smooth = {name: nk.signal_filter(v, sampling_rate=fs, highcut=hc, method="butterworth", order=4)
              for name, v in clean.items()}
    ref = ref_lead if ref_lead in clean else max(clean, key=lambda n: np.ptp(clean[n]))

    _, info = nk.ecg_peaks(smooth[ref], sampling_rate=fs)
    r = np.asarray(info["ECG_R_Peaks"], dtype=int)
    feats: Dict = {"ref_lead": ref, "n_beats": int(len(r))}
    if len(r) < 3:
        return feats
    rr = np.diff(r) / fs                                   # seconds
    rr_s = float(np.median(rr))
    feats["hr_bpm"] = float(60.0 / rr.mean())
    feats["rr_cv"] = float(rr.std() / rr.mean())           # RR irregularity (AF: irregular RR)
    feats["rmssd_ms"] = float(np.sqrt(np.mean(np.diff(rr) ** 2)) * 1000)

    p_on, quality = _multilead_p_onsets(smooth, r, fs, leads)
    feats["p_detect_rate"] = float(np.mean(np.isfinite(p_on)))   # share of beats with a P onset

    # ---- median beat per lead (window adapted to the heart rate) -------------------
    pre, post = int(min(pre_s, 0.6 * rr_s) * fs), int(min(post_s, 0.9 * rr_s) * fs)
    keep = [i for i, rp in enumerate(r) if rp - pre >= 0 and rp + post <= x.shape[0]]
    if len(keep) < 3:
        feats["quality"] = quality
        return feats
    tm = {name: np.median(np.stack([clean[name][r[i] - pre: r[i] + post] for i in keep]), axis=0)
          for name in leads}
    lp = {name: np.median(np.stack([smooth[name][r[i] - pre: r[i] + post] for i in keep]), axis=0)
          for name in leads}
    t0 = pre                                               # index of the R peak in the window
    on_i, off_i = _qrs_bounds_spatial_velocity(lp, fs, t0, qrs_frac)
    k = max(1, int(round(0.010 * fs)))
    st_i = min(off_i + int(round(st_offset_ms / 1000 * fs)), len(tm[ref]) - 1)

    per_lead, t_ends = {}, []
    for name in leads:
        tmpl = tm[name]
        base = float(tmpl[max(on_i - k + 1, 0): on_i + 1].mean())     # QRS-onset level (10 ms)
        seg = tmpl[on_i: off_i + 1] - base
        rpk = int(np.argmax(seg))
        r_amp, s_amp = float(max(seg.max(), 0.0)), float(min(seg.min(), 0.0))
        # Q wave = initial negative deflection, from QRS onset until the signal returns to the baseline
        # before the first positive deflection; deflections smaller than q_min_mV are treated as noise.
        q_amp, q_dur = 0.0, 0.0
        if r_amp < q_min_mV:                              # QS complex: no positive deflection
            q_amp, q_dur = s_amp, len(seg) / fs * 1000
        else:
            first_pos = int(np.argmax(seg > q_min_mV))
            init = seg[: first_pos + 1]
            if init.min() < -q_min_mV:
                q_amp = float(init.min())
                q_dur = (np.where(init < 0)[0][-1] + 1) / fs * 1000
        p_lo, p_hi = max(0, on_i - int(0.35 * fs)), on_i - int(0.04 * fs)
        p_seg = lp[name][p_lo: p_hi] - float(lp[name][max(on_i - k + 1, 0): on_i + 1].mean()) \
            if p_hi > p_lo else np.zeros(1)
        lead_feats = {
            "st_j_mV": float(tmpl[off_i] - base),
            "st_j60_mV": float(tmpl[st_i] - base),
            "r_amp_mV": r_amp,
            "s_amp_mV": s_amp,
            "net_qrs_mV": r_amp + s_amp,
            "q_amp_mV": q_amp,
            "q_dur_ms": q_dur,
            "q_r_ratio": abs(q_amp) / r_amp if r_amp >= q_min_mV else float("inf"),
            "r_peak_time_ms": rpk / fs * 1000,             # QRS onset to the tallest positive deflection
            "s_dur_ms": float(np.sum(seg[rpk:] < 0) / fs * 1000),
            "p_amp_mV": float(p_seg[np.argmax(np.abs(p_seg))]),   # ~0 in a median beat without P waves
        }
        base_lp = float(lp[name][max(on_i - k + 1, 0): on_i + 1].mean())
        te = _tangent_t_end(lp[name], base_lp, fs, off_i, rr_s)
        if te is not None:
            lead_feats["t_amp_mV"], lead_feats["t_end_ms"] = te[1], (te[0] - t0) / fs * 1000
            t_ends.append(te[0])
        per_lead[name] = lead_feats
    feats["leads"] = per_lead

    p_i = t0 + _nanmedian(p_on[keep] - r[keep])
    feats["qrs_ms"] = (off_i - on_i) / fs * 1000
    feats["pr_ms"] = (on_i - p_i) / fs * 1000
    feats["fiducials_ms_from_r"] = {"p_onset": (p_i - t0) / fs * 1000, "qrs_onset": (on_i - t0) / fs * 1000,
                                    "qrs_offset": (off_i - t0) / fs * 1000}
    if len(t_ends) >= 3:
        t_end = float(np.percentile(t_ends, t_end_quantile))
        feats["fiducials_ms_from_r"]["t_end"] = (t_end - t0) / fs * 1000
        qt_ms = (t_end - on_i) / fs * 1000                  # QRS onset to T end (ICH E14)
        feats["qt_ms"] = qt_ms
        feats["qtc_bazett_ms"] = qt_ms / rr_s ** 0.5         # QT / RR^(1/2), RR in seconds
        feats["qtc_fridericia_ms"] = qt_ms / rr_s ** (1 / 3)  # QT / RR^(1/3)
        quality["t_end_leads"] = len(t_ends)
        quality["t_end_lead_spread_ms"] = float(np.median(np.abs(np.array(t_ends) - np.median(t_ends))) / fs * 1000)
    feats["quality"] = quality
    if "I" in per_lead and "aVF" in per_lead:
        # Frontal QRS axis from the net QRS amplitudes of leads I (0 deg) and aVF (+90 deg). In the hexaxial
        # system aVF = (II + III) / 2 has sqrt(3)/2 the gain of lead I, hence the 2/sqrt(3) factor.
        feats["axis_deg"] = float(np.degrees(np.arctan2(2.0 / np.sqrt(3.0) * per_lead["aVF"]["net_qrs_mV"],
                                                        per_lead["I"]["net_qrs_mV"])))
    return feats


# =====================================================================================
# 3. Rule-based clinical criteria (configurable; 1 mm = 0.1 mV at standard gain)
# =====================================================================================
@dataclass
class Criteria:
    # Kusumoto et al., 2019 (2018 ACC/AHA/HRS bradycardia and conduction delay guideline)
    pr_first_degree_ms: float = 200.0          # first-degree AV block: PR > 200 ms
    qrs_complete_bbb_ms: float = 120.0         # complete RBBB/LBBB: QRS >= 120 ms
    qrs_ivcd_ms: float = 110.0                 # nonspecific IVCD: QRS > 110 ms without BBB morphology
    sinus_brady_bpm: float = 50.0              # sinus bradycardia: rate < 50 bpm (also Ribeiro et al., 2020)
    rbbb_rpt_v1_ms: float = 50.0               # complete RBBB: R-peak time > 50 ms in V1
    lbbb_rpt_v5v6_ms: float = 60.0             # complete LBBB: R-peak time > 60 ms in V5 and V6
    lafb_axis: Tuple[float, float] = (-90.0, -45.0)   # LAFB: axis -45 to -90 degrees, QRS < 120 ms
    lpfb_axis: Tuple[float, float] = (90.0, 180.0)    # LPFB: axis 90 to 180 degrees, QRS < 120 ms
    # Ribeiro et al., 2020 label rule for sinus tachycardia
    sinus_tachy_bpm: float = 100.0
    # Byrne et al., 2023 (2023 ESC ACS guideline): new J-point ST elevation in >= 2 contiguous leads
    ste_other_mV: float = 0.10                 # >= 1 mm in leads other than V2-V3
    ste_v2v3_men_lt40_mV: float = 0.25
    ste_v2v3_men_ge40_mV: float = 0.20
    ste_v2v3_women_mV: float = 0.15
    # Sharma et al., 2017 (international criteria for athletes' ECGs), used here as EXAMPLE thresholds:
    std_mV: float = 0.05                       # ST depression > 0.05 mV in >= 2 leads (Sharma: vs PR segment;
                                               # here vs QRS onset, the same reference as ST elevation)
    std_min_leads: int = 2
    twi_mV: float = 0.10                       # T-wave inversion >= 1 mm in >= 2 contiguous leads ...
    twi_exclude: Tuple[str, ...] = ("aVR", "III", "V1")   # ... excluding aVR, III and V1
    q_r_ratio: float = 0.25                    # pathological Q: Q/R >= 0.25 or duration >= 40 ms
    q_dur_ms: float = 40.0                     # in >= 2 contiguous leads ...
    q_exclude: Tuple[str, ...] = ("III", "aVR")           # ... except III and aVR
    # Lead groups for "contiguous leads": a convention; state the grouping you use in the paper.
    groups: Dict[str, Tuple[str, ...]] = field(default_factory=lambda: {
        "inferior": ("II", "III", "aVF"),
        "lateral": ("I", "aVL", "V5", "V6"),
        "anterior": ("V1", "V2", "V3", "V4"),
    })


def _contiguous(flags: Dict[str, bool], groups: Dict[str, Tuple[str, ...]]) -> bool:
    """True if >= 2 flagged leads fall in the same lead group."""
    return any(sum(flags.get(l, False) for l in g) >= 2 for g in groups.values())


def apply_criteria(f: Dict, c: Criteria = Criteria(), female: Optional[bool] = None,
                   age: Optional[float] = None) -> Dict[str, Optional[bool]]:
    """Evaluate guideline criteria on extracted features. None = not evaluable."""
    out: Dict[str, Optional[bool]] = {}
    hr, pr, qrs, axis = f.get("hr_bpm"), f.get("pr_ms"), f.get("qrs_ms"), f.get("axis_deg")
    L = f.get("leads", {})
    ok = lambda v: v is not None and np.isfinite(v)
    out["sinus_brady_rate"] = hr < c.sinus_brady_bpm if ok(hr) else None
    out["sinus_tachy_rate"] = hr > c.sinus_tachy_bpm if ok(hr) else None
    out["first_degree_avb"] = pr > c.pr_first_degree_ms if ok(pr) else None
    if ok(qrs):
        rpt = lambda lead: L.get(lead, {}).get("r_peak_time_ms", np.nan)
        rbbb = qrs >= c.qrs_complete_bbb_ms and rpt("V1") > c.rbbb_rpt_v1_ms
        lbbb = qrs >= c.qrs_complete_bbb_ms and rpt("V5") > c.lbbb_rpt_v5v6_ms and rpt("V6") > c.lbbb_rpt_v5v6_ms
        out["rbbb_partial_criteria"] = bool(rbbb)
        out["lbbb_partial_criteria"] = bool(lbbb)
        out["ivcd_nonspecific"] = bool(qrs > c.qrs_ivcd_ms and not (rbbb or lbbb))
        if ok(axis):
            out["lafb_axis"] = bool(qrs < c.qrs_complete_bbb_ms and c.lafb_axis[0] <= axis <= c.lafb_axis[1])
            out["lpfb_axis"] = bool(qrs < c.qrs_complete_bbb_ms and c.lpfb_axis[0] <= axis <= c.lpfb_axis[1])
    if L:
        def ste_threshold(lead):
            if lead in ("V2", "V3"):
                if female is None or (not female and age is None):
                    return None
                if female:
                    return c.ste_v2v3_women_mV
                return c.ste_v2v3_men_lt40_mV if age < 40 else c.ste_v2v3_men_ge40_mV
            return c.ste_other_mV
        ste = {}
        for lead, v in L.items():
            th = ste_threshold(lead)
            if th is not None:
                ste[lead] = v["st_j_mV"] >= th
        out["stemi_st_criteria"] = _contiguous(ste, c.groups)
        out["st_depression"] = sum(v["st_j_mV"] < -c.std_mV for v in L.values()) >= c.std_min_leads
        out["t_inversion"] = _contiguous({l: v.get("t_amp_mV", 0.0) <= -c.twi_mV for l, v in L.items()
                                          if l not in c.twi_exclude}, c.groups)
        out["pathological_q"] = _contiguous(
            {l: (v["q_amp_mV"] < 0) and (v["q_r_ratio"] >= c.q_r_ratio or v["q_dur_ms"] >= c.q_dur_ms)
             for l, v in L.items() if l not in c.q_exclude}, c.groups)
    return out


# =====================================================================================
# 4. Utility metrics
# =====================================================================================
def macro_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Macro-averaged AUC over classes that have both positives and negatives."""
    y_true, y_score = np.atleast_2d(y_true.T).T, np.atleast_2d(y_score.T).T
    aucs = [roc_auc_score(y_true[:, k], y_score[:, k]) for k in range(y_true.shape[1])
            if 0 < y_true[:, k].sum() < len(y_true)]
    return float(np.mean(aucs)) if aucs else float("nan")


def _valid(yb: np.ndarray) -> bool:
    s = yb.sum(axis=0)
    return bool(((s > 0) & (s < len(yb))).all())


def bootstrap_ci(y_true, y_score, n_boot=1000, alpha=0.05, seed=0, max_tries=100000):
    """Percentile bootstrap CI of macro AUC on a fixed test set. Resamples that leave a class without
    positives are redrawn, as in Strodthoff et al. (2021)."""
    y_true, y_score = np.atleast_2d(y_true.T).T, np.atleast_2d(y_score.T).T
    rng, vals, tries = np.random.default_rng(seed), [], 0
    while len(vals) < n_boot and tries < max_tries:
        tries += 1
        i = rng.integers(0, len(y_true), len(y_true))
        if _valid(y_true[i]):
            vals.append(macro_auc(y_true[i], y_score[i]))
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return macro_auc(y_true, y_score), float(lo), float(hi)


def paired_delta_ci(y_true, score_trtr, score_tstr, n_boot=1000, alpha=0.05, seed=0, max_tries=100000):
    """Delta = AUC(TRTR) - AUC(TSTR) with a paired bootstrap (same resampled test records for both)."""
    y_true = np.atleast_2d(y_true.T).T
    score_trtr, score_tstr = np.atleast_2d(score_trtr.T).T, np.atleast_2d(score_tstr.T).T
    rng, vals, tries = np.random.default_rng(seed), [], 0
    while len(vals) < n_boot and tries < max_tries:
        tries += 1
        i = rng.integers(0, len(y_true), len(y_true))
        if _valid(y_true[i]):
            vals.append(macro_auc(y_true[i], score_trtr[i]) - macro_auc(y_true[i], score_tstr[i]))
    point = macro_auc(y_true, score_trtr) - macro_auc(y_true, score_tstr)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(point), float(lo), float(hi)


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


def delong_test(y_true: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> Dict[str, float]:
    """DeLong et al. (1988) test for two correlated AUCs on the same binary labels."""
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
    z = (aucs[0] - aucs[1]) / np.sqrt(var)
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]), "z": float(z),
            "p": float(2 * stats.norm.sf(abs(z)))}


def relative_utility(auc_tstr: float, auc_trtr: float) -> float:
    """Share of real-data performance retained when training on synthetic data."""
    return auc_tstr / auc_trtr


def non_inferior(delta_ci_upper: float, margin: float) -> bool:
    """TSTR is non-inferior to TRTR if the upper CI bound of AUC(TRTR) - AUC(TSTR) is below the margin."""
    return delta_ci_upper < margin


def feature_fidelity(real: Iterable[float], synth: Iterable[float], tolerance: Optional[float] = None,
                     n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> Dict:
    """Compare one clinical feature (e.g., PR in ms within class CD) between real and synthetic ECGs.
    Equivalence decision: the bootstrap CI of the median difference (synthetic - real) lies entirely
    within +/- tolerance. The KS p-value is descriptive only (with thousands of ECGs it is almost
    always small)."""
    a = np.asarray(list(real), float)
    b = np.asarray(list(synth), float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    ks = stats.ks_2samp(a, b)
    rng = np.random.default_rng(seed)
    diffs = [np.median(rng.choice(b, len(b))) - np.median(rng.choice(a, len(a))) for _ in range(n_boot)]
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    res = {"n_real": len(a), "n_synth": len(b), "median_real": float(np.median(a)),
           "median_synth": float(np.median(b)), "median_diff": float(np.median(b) - np.median(a)),
           "median_diff_ci": (float(lo), float(hi)),
           "wasserstein": float(stats.wasserstein_distance(a, b)), "ks_stat": float(ks.statistic),
           "ks_p": float(ks.pvalue)}
    if tolerance is not None:
        res["within_tolerance"] = bool(-tolerance <= lo and hi <= tolerance)
    return res


def criterion_rate_gap(real_flags: Iterable[Optional[bool]], synth_flags: Iterable[Optional[bool]],
                       margin: Optional[float] = None, n_boot=1000, alpha=0.05, seed=0) -> Dict:
    """Difference in the share of ECGs meeting a criterion (synthetic minus real), with a bootstrap CI.
    Flags that are None (criterion not evaluable, see apply_criteria) are dropped and counted.
    With a margin (e.g. 0.05 = 5 percentage points), 'equivalent' is True if the CI lies within +/- margin."""
    def clean(flags):
        flags = list(flags)
        kept = np.asarray([float(v) for v in flags if v is not None], float)
        return kept[np.isfinite(kept)], len(flags) - int(np.isfinite(kept).sum())
    (a, a_drop), (b, b_drop) = clean(real_flags), clean(synth_flags)
    rng = np.random.default_rng(seed)
    d = [rng.choice(b, len(b)).mean() - rng.choice(a, len(a)).mean() for _ in range(n_boot)]
    lo, hi = np.quantile(d, [alpha / 2, 1 - alpha / 2])
    res = {"n_real": len(a), "n_synth": len(b), "not_evaluable_real": a_drop, "not_evaluable_synth": b_drop,
           "rate_real": float(a.mean()), "rate_synth": float(b.mean()), "gap": float(b.mean() - a.mean()),
           "ci": (float(lo), float(hi))}
    if margin is not None:
        res["equivalent"] = bool(-margin <= lo and hi <= margin)
    return res


def measurement_noise(alg_a: Iterable[float], alg_b: Iterable[float], q: float = 0.5) -> float:
    """Quantile of |difference| between two measurement algorithms on the SAME real records (for example
    two algorithms in PTB-XL+). Usable as an empirical tolerance for feature fidelity."""
    d = np.abs(np.asarray(list(alg_a), float) - np.asarray(list(alg_b), float))
    return float(np.quantile(d[np.isfinite(d)], q))


def scorecard(task: Task, y_test: np.ndarray, score_trtr: np.ndarray, score_tstr: np.ndarray,
              margin: float, n_boot: int = 1000, seed: int = 0) -> Dict:
    """Task-based utility for one task: TRTR and TSTR macro AUC with CIs, delta, relative utility,
    and a non-inferiority decision against a pre-specified margin."""
    trtr = bootstrap_ci(y_test, score_trtr, n_boot=n_boot, seed=seed)
    tstr = bootstrap_ci(y_test, score_tstr, n_boot=n_boot, seed=seed)
    delta = paired_delta_ci(y_test, score_trtr, score_tstr, n_boot=n_boot, seed=seed)
    return {"task": task.name, "use_case": task.use_case, "requires": task.requires,
            "auc_trtr": trtr, "auc_tstr": tstr, "delta": delta,
            "relative_utility": relative_utility(tstr[0], trtr[0]),
            "margin": margin, "non_inferior": non_inferior(delta[2], margin)}


# =====================================================================================
# Demo / smoke test
# =====================================================================================
if __name__ == "__main__":
    import sys

    print("1) Features on a real 12-lead ECG (pass a WFDB record path, e.g. records500/00000/00001_hr)")
    if len(sys.argv) > 1:
        import wfdb
        rec = wfdb.rdrecord(sys.argv[1])
        names = [normalize_lead(n) for n in rec.sig_name]
        use = [i for i, n in enumerate(names) if n in LEADS_12]
        f = extract_features(rec.p_signal[:10 * rec.fs, use], [names[i] for i in use], fs=rec.fs)
        g = lambda k: f.get(k, float("nan"))
        print(f"  HR {g('hr_bpm'):.1f} bpm, PR {g('pr_ms'):.0f} ms, QRS {g('qrs_ms'):.0f} ms, QT {g('qt_ms'):.0f} ms, "
              f"QTcF {g('qtc_fridericia_ms'):.0f} ms, axis {g('axis_deg'):.0f} deg")
        print("  criteria:", apply_criteria(f))
    else:
        print("  (skipped: no record path given)")

    print("2) Labels")
    scp = pd.Series(["{'NORM': 100.0, 'SR': 0.0}", "{'IMI': 80.0, 'SR': 0.0}", "{'CLBBB': 100.0, 'AFIB': 0.0}",
                     "{'NORM': 50.0, 'LAFB': 100.0, 'SR': 0.0}", "{'SR': 0.0}"])
    for name in ("acs_superclass", "triage_ribeiro6", "bench_superdiagnostic"):
        print(" ", name, TASKS[name].classes, build_labels(scp, TASKS[name]).tolist())
    print("  triage_norm", build_triage_labels(scp).tolist())

    print("3) Utility metrics on simulated classifier scores")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(800, 5))
    s_real = y + rng.normal(0, 0.8, y.shape)
    s_syn = y + rng.normal(0, 1.1, y.shape)
    card = scorecard(TASKS["bench_superdiagnostic"], y, s_real, s_syn, margin=0.02, n_boot=300)
    print("  TRTR", np.round(card["auc_trtr"], 3), "TSTR", np.round(card["auc_tstr"], 3),
          "delta", np.round(card["delta"], 3), "RU", round(card["relative_utility"], 3),
          "non-inferior", card["non_inferior"])
    print("  DeLong (class 0):", {k: round(v, 4) for k, v in delong_test(y[:, 0], s_real[:, 0], s_syn[:, 0]).items()})
    r2 = lambda v: round(v, 2) if isinstance(v, float) else (tuple(round(x, 2) for x in v) if isinstance(v, tuple) else v)
    print("  PR fidelity:", {k: r2(v) for k, v in
                             feature_fidelity(rng.normal(160, 20, 500), rng.normal(166, 25, 500), tolerance=10).items()})
