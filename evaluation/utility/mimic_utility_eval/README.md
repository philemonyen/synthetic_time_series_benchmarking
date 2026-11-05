# MIMIC-IV synthetic-data utility evaluation

Code for the utility tests in our review *Real-World Use Cases Testable with Our Preprocessed MIMIC-IV ICU
Time Series* (Sections 4–5). It does for MIMIC-IV what `ecg_use_case_utility.py` does for PTB-XL.

| File | What it is |
|---|---|
| `mimic_utility.py` | The module: labels, tasks, Tier 1–3 utility tests, privacy checks, report, command line |
| `test_mimic_utility.py` | Assertion tests; `--real` also reproduces the review's numbers on our files |
| `validate_pipeline.py` | Positive and negative control generators with known verdicts |
| `paths.py` | Where the real files live (edit for your machine or the cluster) |

Dependencies: numpy, scipy, scikit-learn. No GPU needed.

## Pipeline

```
real train ──► generator ──► synthetic stays (N x 16 x 72, original units, masks)
                                   │
real train + real test ────────────┤
                                   ▼
   0. prepare_synthetic   layout check, mask binarization, unit sanity check
      ValueGrid           resolution matching: snap synthetic values to the real value grid
   1. labels              same rule-based labels on real and synthetic stays (measured slots only)
   2. real_reference      once per dataset: held-out real patients play a perfect generator
   3. Tier 1              TSTR vs TRTR on real test (non-inferiority, patient-level bootstrap)
   4. Tier 2              fidelity on measured slots, each metric next to the real-vs-real reference
   5. Tier 3              clinical-event rates (equivalence) and outcome gradients
   6. Privacy             DCR, membership inference, NNAA (reported separately from utility)
                                   ▼
                    report.json + report.md
```

## Running it

One generator from the command line:

```bash
python mimic_utility.py \
  --train train.npy --test test.npy \
  --stays-train stays_train.npy --stays-test stays_test.npy \
  --subjects-train subjects_train.npy --subjects-test subjects_test.npy \
  --annotations annotations.json --channels channels.json \
  --synthetic sssd_mimic.npy \
  --out reports/sssd
```

The first run writes `reports/sssd_reference.json`. Pass it to later runs with `--reference` so every
generator is judged against the same real-vs-real reference without recomputing it.

All four generators in Python:

```python
import mimic_utility as M
tr = M.load_real("train.npy", "stays_train.npy", "subjects_train.npy", "annotations.json", "real_train")
te = M.load_real("test.npy", "stays_test.npy", "subjects_test.npy", "annotations.json", "real_test")
syn = {name: M.load_synthetic(f"{name}.npy")[0] for name in ["sssd_ecg", "tts_gan", "transfusion", "diffusion_ts"]}
reports = M.evaluate_many(tr, te, syn, M.load_channels("channels.json"))
for name, rep in reports.items():
    open(f"{name}.md", "w").write(M.report_markdown(rep))
```

Runtime, measured on a 2-core machine: about 3 minutes for the reference, and about 7 minutes per generator
for 10 tasks with 500 bootstrap samples. Expect roughly twice that for all default tasks with 1,000 samples;
use `n_boot=300` while iterating.

Threads: scikit-learn's gradient boosting uses OpenMP. When several evaluations share a machine (for example
one job per generator on a cluster node), set `OMP_NUM_THREADS` so the total does not exceed the cores. In
our checks, oversubscription turned a 35-second run into more than 20 minutes.

## What the generator must deliver

- **Shape.** N x 16 x 72 in the channel order of `channels.json` (value, mask, value, mask, …).
  N x 72 x 16 is transposed automatically.
- **Units.** Original units, de-normalized. A median heart rate outside 30–200 bpm triggers a warning.
- **Masks.** Generate the mask channels too; continuous masks are thresholded at 0.5. Values-only output
  (N x 8 x 72) is accepted, but the labs are then treated as hourly measurements, the missingness results
  are meaningless, and the report says so.
- **Training data.** Train the generator on `train.npy` only. The test split must never be seen.
- **Labels (optional).** For a conditional generator, an `.npz` with arrays such as `mortality` (one value per
  synthetic stay) passed as `--synthetic-labels`. Without labels the outcome tasks are skipped and
  listed in the report.

## Resolution matching (on by default)

Real measured values sit on a grid. In our training data, 92–99.9% of measured vital signs, glucose and
sodium values are integers, creatinine moves in 0.1 mg/dL steps, and 5.9% of all sodium values are exactly
135 mmol/L. Generators output continuous values, so a rule like *sodium < 135* would count values such as
134.97 that never occur in real data.

We measured the effect: adding noise of just 0.001 to the real data shifted the event rates by up to 5
percentage points (sodium < 135 by 5.0, SpO₂ < 90% by 2.8). It also let a classifier tell copies from
originals by their decimals alone.

So before computing labels and metrics, `evaluate()` snaps every synthetic value to the nearest value
observed in the real training data for that signal. Values outside the observed range stay as they are.
Range and validity checks are also reported on the raw output, and the mean shift per signal is in the
report. Turn this off with `match_resolution=False` or `--no-resolution-matching`.

## Reading the report

**Tier 1 (task utility).** For each task, a model is trained on real (TRTR) and on synthetic (TSTR)
stays, size-matched, and both are tested on the real test stays. Delta = AUROC(TRTR) − AUROC(TSTR), with a
95% CI from resampling test *patients*.

| Decision | Meaning |
|---|---|
| non-inferior | upper CI bound of delta < margin (default 0.02 AUROC; forecasting: 5% of TRTR MAE) |
| inferior | lower CI bound > margin |
| inconclusive | anything else |

The *perfect-generator delta* column repeats the test with held-out real patients in place of synthetic
data. If even that cannot pass (UNDERPOWERED), the task cannot show non-inferiority at this margin with
our test set: widen the margin before seeing results, or drop the task. The seed and model-family spread
of real models, the anchor proposed in the ECG review, is in the JSON for each task.

Default tasks:

| Task | Label (computed from the series) | Input |
|---|---|---|
| `hypotension_h49_72` | derived MAP < 65 mmHg ≥ 2 h in hours 49–72 | hours 0–48 |
| `shock_index_h49_72` | HR/SBP ≥ 1.0 ≥ 2 h in hours 49–72 | hours 0–48 |
| `spo2_lt90_h49_72` | SpO₂ < 90% ≥ 2 h in hours 49–72 | hours 0–48 |
| `tachycardia_h49_72` | HR > 100 ≥ 2 h in hours 49–72 | hours 0–48 |
| `aki_new_h25_72` | new KDIGO creatinine AKI after hour 24 | hours 0–24 |
| `crossing_<signal>` | threshold crossing in the next hour (HR > 100, SBP < 90, RR > 24, SpO₂ < 92) | previous 12 h |
| `forecast_<signal>` | next-hour value (MAE; last value carried forward as a baseline) | previous 6 h |
| `outcome_*` | mortality, ICU mortality, 1-year mortality, ICU LOS > 7 d, discharge home, ICD sepsis / heart failure / AKI | hours 0–72, needs synthetic labels |

Subgroup results (race, insurance, sex, age band) for each task are in the JSON.

**Tier 2 (fidelity, measured slots only).**
- Marginals: Wasserstein distance per IQR, KS, quantile differences, MDD, skewness and kurtosis
  differences.
- Dynamics: Wasserstein distance of hour-to-hour changes, and ACD over lags 1–24 h.
- Relations between signals: cross-signal correlation difference.
- Missingness: number, first hour and spacing of measurements.
- Validity: DBP ≥ SBP share, values outside or at the clip bounds, measured values equal to the
  fill constants.
- Detection and sample-level scores: discriminative score, α-precision, β-recall and authenticity.

Every metric is shown next to the reference (real test vs real train). A perfect generator scores
about the reference, not 0. The β-recall integral (IR-β) is not calibrated in high dimensions, so read the
sample-level scores against the reference.

**Tier 3 (clinical-event consistency).** Rates of 13 events over hours 0–72 (hypotension, shock index,
SpO₂, heart rate, AKI, glucose and sodium thresholds). Each rate is compared with the real training rate:
equivalent if the 95% CI of the gap lies within ±0.05. With synthetic mortality labels, the report also
compares how much more often each event occurs in deaths than in survivors.

**Privacy (not part of utility).** All checks use a fixed embedding: per-stay summaries, and the
forward-filled vital-sign trajectories. Nearest neighbours are searched in the *full* training set and the
*full* synthetic set, because an attacker sees the whole release; only the query points are sampled.
- DCR: distance from synthetic stays to the closest training stay, compared with held-out real stays.
- Membership inference: AUROC for telling training patients from held-out patients by their distance to
  the nearest synthetic stay (Chen et al., 2020).
- NNAA privacy loss (Yale et al., 2019), on balanced samples as defined.

Expected for no memorization: share below the hold-out 5th percentile ≈ 0.05, MIA AUROC ≈ 0.5, privacy
loss ≈ 0.

## Validation

`python test_mimic_utility.py --real` checks the following (about 50 seconds):
- every label rule, on hand-built stays;
- every statistic, against known answers;
- the full runner, on toy cohorts;
- the review's numbers on our files: all 13 event rates, both AKI baseline variants and the sodium grades
  are reproduced exactly.

`python validate_pipeline.py` splits the real training patients into halves A and B. A plays the
generator's training data, and seven control "generators" are built from B or A. The full pipeline then
runs on each (10 tasks, 500 bootstrap samples, about 2.5 minutes per control on 2 cores). Each control is
caught by the check designed for it, and the perfect generator passes:

| Control | What it breaks | Tier 1: non-inferior / inconclusive / inferior | Caught by (synthetic vs real-vs-real reference) | MIA AUROC |
|---|---|---|---|---|
| perfect (held-out real patients) | nothing | 9 / 1 / 0; the 1 is SpO₂ < 90%, flagged underpowered | every fidelity metric at the reference; 13/13 event rates equivalent | 0.49 |
| copy (training stays) | privacy | 10 / 0 / 0 | MIA AUROC 1.00; 100% of stays below the hold-out 5th percentile of DCR; authenticity 0.24 vs 0.74; NNAA loss 0.26 | 1.00 |
| time_shuffled | temporal structure | 1 / 4 / 5 | ACD 0.249 vs 0.003; hourly-change Wasserstein 3.29 vs 0.03; 8/13 event rates | 0.49 |
| decoupled | relations between signals | 7 / 0 / 3 | cross-signal correlation difference 0.079 vs 0.008; derived MAP < 65 rate 0.30 vs 0.48; discriminator AUROC 0.996 | 0.49 |
| hourly_labs | missingness | 8 / 2 / 0 | glucose measurements per stay 72 vs 5; discriminator AUROC 1.00 | 0.49 |
| noisy | marginals and dynamics | 6 / 1 / 3 | Wasserstein/IQR 0.076 vs 0.009; 1/13 event rates equivalent | 0.49 |
| collapsed (300 stays repeated) | diversity | 2 / 0 / 8 | recall at β = 1: 0.17 vs 0.97 | 0.49 |

Two lessons from the controls:
- **No single score is enough.** The discriminator on per-stay summaries barely notices time shuffling
  (AUROC 0.60), because summaries ignore order. That is why the dynamics metrics are reported separately.
  Tier 1 alone would pass the memorizer, which is why privacy is checked separately.
- **A discriminator AUROC clearly below 0.5 is a warning, not a success.** It happens when synthetic stays
  are copies of training stays: each copy has a twin with the opposite label in the classifier's training
  data (0.36 for the copy control). Check privacy.

Full reports for every control are in `controls/` (`<control>.md`, `<control>.json`, `summary.md`).

## Differences from the original methods

- Discriminative score uses gradient boosting on per-stay summaries, not a 2-layer LSTM (Yoon et al., 2019).
- α-precision, β-recall and authenticity use a fixed standardized embedding, not Alaa et al.'s (2022)
  learned one-class embedding. For their implementation see Synthcity (Qian et al., 2023).
- MDD and ACD follow TSGBench (Ang et al., 2023) in spirit. Values are pooled over time per signal, and
  the ACF is computed on measured hour pairs after removing each stay's mean.
- Threshold-crossing tasks follow Esteban et al. (2017) in spirit; the thresholds are ours.

## References for the methods

- Alaa, A., van Breugel, B., Saveliev, E. S., & van der Schaar, M. (2022). How faithful is your synthetic data? Sample-level metrics for evaluating and auditing generative models. *ICML*, PMLR 162, 290–306.
- Ang, Y., Huang, Q., Bao, Y., et al. (2023). TSGBench: time series generation benchmark. *PVLDB*, 17(3), 305–318.
- Chen, D., Yu, N., Zhang, Y., & Fritz, M. (2020). GAN-Leaks: a taxonomy of membership inference attacks against generative models. *ACM CCS 2020*. doi:10.1145/3372297.3417238
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach. *Biometrics*, 44(3), 837–845.
- Esteban, C., Hyland, S. L., & Rätsch, G. (2017). Real-valued (medical) time series generation with recurrent conditional GANs. arXiv:1706.02633.
- Qian, Z., Davis, R., & van der Schaar, M. (2023). Synthcity: a benchmark framework for diverse use cases of tabular synthetic data. *NeurIPS Datasets and Benchmarks*, 3173–3188.
- Yale, A., Dash, S., Dutta, R., Guyon, I., Pavao, A., & Bennett, K. P. (2019). Privacy preserving synthetic health data. *ESANN 2019*, 465–470.
- Yoon, J., Jarrett, D., & van der Schaar, M. (2019). Time-series generative adversarial networks. *NeurIPS 2019*.
- Clinical definitions (KDIGO 2012; Evans et al., 2021; Koch et al., 2019; ADA 2025; Spasovski et al., 2014; Funk et al., 2010; Egi et al., 2006): see the review's reference list.
