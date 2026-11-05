"""Validate mimic_utility.py with control "generators" whose correct verdicts are known in advance.

The real training split is divided by patient into A and B. A plays the generator's training data; the
"synthetic" sets are built from B (patients the generator never saw) or from A itself:

  perfect        B unchanged: a fresh sample of real patients. Should pass every utility check, sit at the
                 reference on fidelity, and show no privacy signal.
  copy           A plus negligible jitter: memorized training stays. Utility passes, privacy must fail.
  time_shuffled  B with the 72 hours of each stay permuted (all channels together): marginals and same-hour
                 correlations intact, temporal structure destroyed.
  decoupled      B with each signal taken from a different random stay: marginals and each signal's own
                 dynamics intact, relations between signals destroyed.
  hourly_labs    B with glucose, creatinine and sodium forward-filled and marked as measured every hour:
                 values intact, missingness pattern wrong.
  noisy          B plus Gaussian noise of 0.5 SD on every measured value: marginals widened, dynamics noisy.
  collapsed      300 stays of B repeated to the full size: realistic stays, no diversity.

Run: python validate_pipeline.py [--n-boot 500] [--controls perfect copy ...] [--out-dir controls]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mimic_utility as M  # noqa: E402
from paths import PATHS as P  # noqa: E402

TASKS = ["hypotension_h49_72", "shock_index_h49_72", "spo2_lt90_h49_72", "aki_new_h25_72", "crossing_sbp",
         "crossing_heart_rate", "forecast_heart_rate", "forecast_sbp", "outcome_mortality", "outcome_icd_aki"]


def time_shuffled(X, rng):
    idx = np.argsort(rng.random((len(X), M.N_T)), axis=1)
    return X[np.arange(len(X))[:, None, None], np.arange(M.N_CH)[None, :, None], idx[:, None, :]]


def decoupled(X, rng):
    Y = X.copy()
    for i in range(len(M.SIG)):
        Y[:, 2 * i:2 * i + 2] = X[rng.permutation(len(X)), 2 * i:2 * i + 2]
    return Y


def hourly_labs(X, rng=None):
    Y = X.copy()
    for s in M.LABS:
        i = M.SIG.index(s)
        v, m = Y[:, 2 * i], Y[:, 2 * i + 1] > 0.5
        first = M.first_value({s: v}, {s: m}, s)
        idx = np.where(m, np.arange(M.N_T), 0)
        np.maximum.accumulate(idx, axis=1, out=idx)
        ff = v[np.arange(len(v))[:, None], idx]
        before = ~np.maximum.accumulate(m, axis=1)
        ff[before] = np.repeat(first[:, None], M.N_T, axis=1)[before]
        ok = np.isfinite(first)
        Y[ok, 2 * i], Y[ok, 2 * i + 1] = ff[ok], 1.0
    return Y


def noisy(X, rng, scale=0.5, bounds=M.DEFAULT_CLIP):
    Y = X.copy()
    v, m = M.split(X)
    for i, s in enumerate(M.SIG):
        sd = v[s][m[s]].std()
        Y[:, 2 * i] = np.clip(np.where(m[s], v[s] + rng.normal(0, scale * sd, v[s].shape), v[s]), *bounds[s])
    return Y


def copy_jitter(X, rng, bounds=M.DEFAULT_CLIP):
    """Memorized training stays with negligible jitter, clipped to the bounds like the preprocessing."""
    Y = X.copy()
    for i, s in enumerate(M.SIG):
        Y[:, 2 * i] = np.clip(Y[:, 2 * i] + rng.normal(0, 1e-3, Y[:, 2 * i].shape), *bounds[s])
    return Y


def build_controls(A: M.Cohort, B: M.Cohort, names, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for name in names:
        if name == "perfect":
            c = B.subset(np.arange(len(B)))
        elif name == "copy":
            c = A.subset(np.arange(len(A)))
            c.X = copy_jitter(c.X, rng)
        elif name == "collapsed":
            c = B.subset(np.resize(rng.choice(len(B), 300, replace=False), len(B)))
        else:
            c = B.subset(np.arange(len(B)))
            c.X = {"time_shuffled": time_shuffled, "decoupled": decoupled, "hourly_labs": hourly_labs, "noisy": noisy}[name](c.X, rng)
        c.subjects, c.groups, c.name = None, {}, name      # a generator outputs no patient ids or attributes
        out[name] = c
    return out


def summarize(name, r):
    t1 = [x for x in r["tier1"]]
    dec = [x["decision"] for x in t1]
    t2, t3, pv = r["tier2"], r["tier3"], r["privacy"]["raw_vitals"]
    sig_s, sig_r = t2["signals"]["synthetic"], t2["signals"]["reference"]
    vit = lambda key, d: float(np.mean([d[s][key] for s in M.VITALS]))
    return {"control": name,
            "tier1_non_inferior": dec.count("non-inferior"), "tier1_inconclusive": dec.count("inconclusive"),
            "tier1_inferior": dec.count("inferior"), "tier1_n": len(dec),
            "mean_delta_auroc": float(np.mean([x["delta_auroc"][0] for x in t1 if x["kind"] != "forecast"])),
            "w_per_iqr_vitals": vit("wasserstein_per_iqr", sig_s), "w_per_iqr_vitals_ref": vit("wasserstein_per_iqr", sig_r),
            "acd_vitals": vit("acd", sig_s), "acd_vitals_ref": vit("acd", sig_r),
            "change_w_vitals": vit("change_wasserstein", sig_s), "change_w_vitals_ref": vit("change_wasserstein", sig_r),
            "corr_diff": t2["cross_correlation"]["mean_abs_diff_synthetic"], "corr_diff_ref": t2["cross_correlation"]["mean_abs_diff_reference"],
            "glucose_median_count": t2["missingness"]["synthetic"]["glucose"]["median_count_other"],
            "discriminative_auroc": t2["discriminative"]["synthetic"]["auroc"], "discriminative_auroc_ref": t2["discriminative"]["reference"]["auroc"],
            "recall_at_beta1": t2["sample_level"]["synthetic"]["recall_at_beta1"], "recall_at_beta1_ref": t2["sample_level"]["reference"]["recall_at_beta1"],
            "authenticity": t2["sample_level"]["synthetic"]["authenticity"], "authenticity_ref": t2["sample_level"]["reference"]["authenticity"],
            "tier3_equivalent": sum(e["equivalent"] for e in t3["events"].values()), "tier3_n": len(t3["events"]),
            "mia_auroc": pv["mia_auroc"], "privacy_loss": pv["privacy_loss"], "share_below_p5": pv["share_synth_below_holdout_p5"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", nargs="*", default=["perfect", "copy", "time_shuffled", "decoupled", "hourly_labs", "noisy", "collapsed"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", default="controls")
    a = ap.parse_args()
    out = Path(a.out_dir); out.mkdir(exist_ok=True)
    channels = M.load_channels(P["channels"])
    tr = M.load_real(P["train"], P["stays_train"], P["subjects_train"], P["annotations"], "real_train")
    te = M.load_real(P["test"], P["stays_test"], P["subjects_test"], P["annotations"], "real_test")
    A, B = M.split_by_patient(tr, seed=42)
    print(f"A {len(A)} stays (generator training data), B {len(B)} stays (held out)", flush=True)
    ref_path = out / "reference.json"
    if ref_path.exists():
        ref = json.load(open(ref_path))
    else:
        ref = M._jsonable(M.real_reference(A, te, TASKS, a.n_boot, verbose=True))
        json.dump(ref, open(ref_path, "w"), indent=1)
    rows = []
    for name, syn in build_controls(A, B, a.controls).items():
        print(f"=== {name}", flush=True)
        rep = M.evaluate(A, te, syn, channels, TASKS, n_boot=a.n_boot, reference=ref, verbose=True)
        json.dump(rep, open(out / f"{name}.json", "w"), indent=1)
        open(out / f"{name}.md", "w").write(M.report_markdown(rep))
        rows.append(summarize(name, rep))
        json.dump(rows, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
