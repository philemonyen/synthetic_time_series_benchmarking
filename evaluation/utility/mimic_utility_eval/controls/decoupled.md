# Synthetic MIMIC-IV utility report

Real train 10876 stays, real test 7338, synthetic 10984. AUROC margin 0.02, forecast margin 5% of TRTR MAE, event-rate margin +/- 0.05. Runtime 131.8 s.
Synthetic values were snapped to the real value grid before labels and metrics (resolution matching). Decision: non-inferior = upper CI bound of (TRTR - TSTR) below the margin; inferior = lower bound above it; otherwise inconclusive. 'Perfect-generator delta' trains on held-out real patients instead of synthetic data; UNDERPOWERED marks tasks where even that cannot pass.

## Tier 1: task utility (TSTR vs TRTR, real test set)

| Task | Prevalence real / synth | AUROC TRTR | AUROC TSTR | Delta (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| hypotension_h49_72 | 0.240 / 0.145 | 0.837 | 0.832 | 0.005 (0.001 to 0.009) | 0.020 | non-inferior | 0.005 (-0.002 to 0.011) |
| shock_index_h49_72 | 0.235 / 0.236 | 0.893 | 0.892 | 0.000 (-0.003 to 0.004) | 0.020 | non-inferior | 0.002 (-0.003 to 0.007) |
| spo2_lt90_h49_72 | 0.036 / 0.038 | 0.850 | 0.847 | 0.003 (-0.013 to 0.018) | 0.020 | non-inferior | 0.022 (-0.002 to 0.046) UNDERPOWERED |
| aki_new_h25_72 | 0.213 / 0.232 | 0.737 | 0.697 | 0.040 (0.026 to 0.054) | 0.020 | inferior | 0.005 (-0.007 to 0.018) |
| crossing_sbp | 0.057 / 0.053 | 0.887 | 0.883 | 0.004 (0.001 to 0.007) | 0.020 | non-inferior | -0.002 (-0.005 to 0.001) |
| crossing_heart_rate | 0.208 / 0.205 | 0.957 | 0.956 | 0.001 (-0.000 to 0.001) | 0.020 | non-inferior | -0.000 (-0.001 to 0.001) |
| outcome_mortality | 0.178 / 0.180 | 0.763 | 0.467 | 0.295 (0.267 to 0.323) | 0.020 | inferior | 0.002 (-0.010 to 0.014) |
| outcome_icd_aki | 0.439 / 0.459 | 0.884 | 0.485 | 0.400 (0.382 to 0.413) | 0.020 | inferior | -0.005 (-0.010 to -0.001) |

| Forecast task | MAE TRTR | MAE TSTR | MAE last value | Delta MAE (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| forecast_heart_rate | 5.26 | 5.27 | 5.56 | 0.00 (-0.01 to 0.01) | 0.26 | non-inferior | -0.02 (-0.04 to -0.01) |
| forecast_sbp | 9.68 | 9.70 | 11.02 | 0.02 (0.00 to 0.04) | 0.48 | non-inferior | -0.02 (-0.04 to -0.01) |

Subgroup with the largest TRTR - TSTR delta for each task (all subgroups are in the JSON):

| Task | Subgroup | Test stays | AUROC TRTR | AUROC TSTR | Delta (95% CI) |
|---|---|---|---|---|---|
| hypotension_h49_72 | age_band = <50 | 1219 | 0.825 | 0.809 | 0.016 (0.007 to 0.026) |
| shock_index_h49_72 | race_group = Hispanic/Latino | 240 | 0.917 | 0.906 | 0.011 (-0.011 to 0.033) |
| spo2_lt90_h49_72 | age_band = >=80 | 1271 | 0.881 | 0.828 | 0.053 (0.009 to 0.110) |
| aki_new_h25_72 | race_group = Other or unknown | 1176 | 0.749 | 0.693 | 0.056 (0.019 to 0.091) |
| crossing_sbp | insurance_group = Other or missing | 1003 | 0.910 | 0.901 | 0.010 (-0.007 to 0.032) |
| crossing_heart_rate | age_band = >=80 | 4966 | 0.947 | 0.945 | 0.002 (0.001 to 0.004) |
| outcome_mortality | race_group = Asian | 216 | 0.799 | 0.462 | 0.337 (0.202 to 0.462) |
| outcome_icd_aki | race_group = Asian | 216 | 0.874 | 0.414 | 0.459 (0.352 to 0.547) |

## Tier 2: fidelity on measured slots (synthetic vs real train; reference = real test vs real train)

| Signal | Wasserstein / IQR | ref | MDD | ref | Hourly change W | ref | ACD | ref | Median count synth / real |
|---|---|---|---|---|---|---|---|---|---|
| heart_rate | 0.013 | 0.007 | 0.0005 | 0.0004 | 0.07 | 0.04 | 0.003 | 0.002 | 71 / 71 |
| sbp | 0.010 | 0.007 | 0.0003 | 0.0003 | 0.03 | 0.03 | 0.002 | 0.002 | 70 / 70 |
| dbp | 0.010 | 0.011 | 0.0003 | 0.0003 | 0.03 | 0.04 | 0.004 | 0.003 | 70 / 70 |
| resp_rate | 0.011 | 0.011 | 0.0003 | 0.0003 | 0.01 | 0.01 | 0.004 | 0.003 | 70 / 70 |
| spo2 | 0.003 | 0.006 | 0.0002 | 0.0003 | 0.01 | 0.01 | 0.003 | 0.003 | 70 / 70 |
| glucose | 0.016 | 0.028 | 0.0006 | 0.0010 | 6.99 | 16.49 | - | - | 5 / 5 |
| creatinine | 0.028 | 0.013 | 0.0007 | 0.0005 | 0.05 | 0.06 | - | - | 5 / 5 |
| sodium | 0.013 | 0.022 | 0.0006 | 0.0006 | 0.42 | 0.76 | - | - | 6 / 6 |

- Cross-signal correlation, mean absolute difference: synthetic 0.079, reference 0.008
- Discriminative score |accuracy - 0.5|: synthetic 0.472 (AUROC 0.996), reference 0.005 (AUROC 0.500)
- IP-alpha / IR-beta / recall at beta = 1 / authenticity: synthetic 0.907 / 0.693 / 0.602 / 0.921; reference 0.992 / 0.483 / 0.972 / 0.740 (read against the reference, not 1.0)
- DBP >= SBP share: synthetic 0.0130, real 0.0000
- Raw generator output outside the clip bounds: none

## Tier 3: clinical-event rates, hours 0-72 (equivalence margin +/- 0.05)

| Event | Real train | Synthetic | Gap (95% CI) | Reference gap | Equivalent |
|---|---|---|---|---|---|
| map_lt65_2h | 0.480 | 0.301 | -0.179 (-0.192 to -0.164) | 0.010 | NO |
| sbp_lt90_2h | 0.309 | 0.299 | -0.010 (-0.021 to 0.003) | -0.004 | yes |
| spo2_lt90_2h | 0.087 | 0.088 | 0.002 (-0.005 to 0.009) | -0.003 | yes |
| hr_gt100_2h | 0.604 | 0.587 | -0.017 (-0.031 to -0.004) | -0.011 | yes |
| hr_lt50_2h | 0.059 | 0.061 | 0.001 (-0.006 to 0.007) | -0.003 | yes |
| shock_index_ge1_2h | 0.448 | 0.452 | 0.004 (-0.009 to 0.018) | -0.003 | yes |
| aki_creatinine | 0.329 | 0.347 | 0.018 (0.006 to 0.031) | 0.003 | yes |
| glucose_lt70_any | 0.067 | 0.066 | -0.001 (-0.008 to 0.006) | 0.005 | yes |
| glucose_lt54_any | 0.018 | 0.019 | 0.000 (-0.003 to 0.004) | 0.002 | yes |
| glucose_gt180_any | 0.392 | 0.391 | -0.000 (-0.014 to 0.012) | -0.013 | yes |
| sodium_lt135_any | 0.376 | 0.366 | -0.009 (-0.021 to 0.004) | 0.003 | yes |
| sodium_lt125_any | 0.024 | 0.026 | 0.002 (-0.002 to 0.006) | -0.000 | yes |
| sodium_gt145_any | 0.184 | 0.180 | -0.004 (-0.015 to 0.007) | -0.014 | yes |

Outcome gradient for mortality (event rate among deaths minus survivors): map_lt65_2h real 0.148 vs synth -0.016; sbp_lt90_2h real 0.123 vs synth -0.000; spo2_lt90_2h real 0.067 vs synth -0.007; hr_gt100_2h real 0.121 vs synth -0.005; hr_lt50_2h real -0.001 vs synth 0.008; shock_index_ge1_2h real 0.155 vs synth -0.005; aki_creatinine real 0.211 vs synth 0.016; glucose_lt70_any real 0.042 vs synth -0.006; glucose_lt54_any real 0.017 vs synth 0.002; glucose_gt180_any real 0.132 vs synth -0.009; sodium_lt135_any real 0.070 vs synth -0.012; sodium_lt125_any real 0.015 vs synth 0.000; sodium_gt145_any real 0.094 vs synth -0.003

## Privacy (not part of utility)

| Embedding | DCR median synth / holdout | Synth below holdout 5th pct | Exact-copy share | MIA AUROC | NNAA privacy loss |
|---|---|---|---|---|---|
| summary | 4.662 / 4.062 | 0.004 | 0.000 | 0.496 | -0.000 |
| raw_vitals | 15.797 / 15.266 | 0.019 | 0.000 | 0.489 | 0.008 |

Expected for a generator that does not memorize: share below the 5th percentile near 0.05, MIA AUROC near 0.5, privacy loss near 0.