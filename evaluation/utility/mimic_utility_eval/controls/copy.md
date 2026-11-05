# Synthetic MIMIC-IV utility report

Real train 10876 stays, real test 7338, synthetic 10876. AUROC margin 0.02, forecast margin 5% of TRTR MAE, event-rate margin +/- 0.05. Runtime 128.1 s.
Synthetic values were snapped to the real value grid before labels and metrics (resolution matching). Decision: non-inferior = upper CI bound of (TRTR - TSTR) below the margin; inferior = lower bound above it; otherwise inconclusive. 'Perfect-generator delta' trains on held-out real patients instead of synthetic data; UNDERPOWERED marks tasks where even that cannot pass.

## Tier 1: task utility (TSTR vs TRTR, real test set)

| Task | Prevalence real / synth | AUROC TRTR | AUROC TSTR | Delta (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| hypotension_h49_72 | 0.240 / 0.240 | 0.837 | 0.836 | 0.001 (-0.001 to 0.002) | 0.020 | non-inferior | 0.005 (-0.002 to 0.011) |
| shock_index_h49_72 | 0.235 / 0.235 | 0.893 | 0.892 | 0.000 (-0.001 to 0.002) | 0.020 | non-inferior | 0.002 (-0.003 to 0.007) |
| spo2_lt90_h49_72 | 0.036 / 0.036 | 0.850 | 0.850 | 0.000 (-0.000 to 0.000) | 0.020 | non-inferior | 0.022 (-0.002 to 0.046) UNDERPOWERED |
| aki_new_h25_72 | 0.213 / 0.214 | 0.737 | 0.735 | 0.002 (-0.002 to 0.007) | 0.020 | non-inferior | 0.005 (-0.007 to 0.018) |
| crossing_sbp | 0.057 / 0.057 | 0.887 | 0.887 | 0.000 (-0.000 to 0.001) | 0.020 | non-inferior | -0.002 (-0.005 to 0.001) |
| crossing_heart_rate | 0.208 / 0.208 | 0.957 | 0.956 | 0.000 (0.000 to 0.001) | 0.020 | non-inferior | -0.000 (-0.001 to 0.001) |
| outcome_mortality | 0.178 / 0.178 | 0.763 | 0.761 | 0.002 (-0.002 to 0.004) | 0.020 | non-inferior | 0.002 (-0.010 to 0.014) |
| outcome_icd_aki | 0.439 / 0.439 | 0.884 | 0.885 | -0.001 (-0.002 to 0.000) | 0.020 | non-inferior | -0.005 (-0.010 to -0.001) |

| Forecast task | MAE TRTR | MAE TSTR | MAE last value | Delta MAE (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| forecast_heart_rate | 5.26 | 5.26 | 5.56 | 0.00 (0.00 to 0.00) | 0.26 | non-inferior | -0.02 (-0.04 to -0.01) |
| forecast_sbp | 9.68 | 9.69 | 11.02 | 0.00 (0.00 to 0.01) | 0.48 | non-inferior | -0.02 (-0.04 to -0.01) |

Subgroup with the largest TRTR - TSTR delta for each task (all subgroups are in the JSON):

| Task | Subgroup | Test stays | AUROC TRTR | AUROC TSTR | Delta (95% CI) |
|---|---|---|---|---|---|
| hypotension_h49_72 | race_group = Asian | 216 | 0.863 | 0.857 | 0.006 (-0.005 to 0.018) |
| shock_index_h49_72 | race_group = Hispanic/Latino | 240 | 0.917 | 0.912 | 0.005 (-0.004 to 0.015) |
| spo2_lt90_h49_72 | race_group = Black | 817 | 0.847 | 0.846 | 0.000 (0.000 to 0.001) |
| aki_new_h25_72 | race_group = Other or unknown | 1176 | 0.749 | 0.738 | 0.011 (-0.002 to 0.023) |
| crossing_sbp | insurance_group = Other or missing | 1003 | 0.910 | 0.908 | 0.003 (-0.001 to 0.008) |
| crossing_heart_rate | insurance_group = Other or missing | 1016 | 0.970 | 0.969 | 0.001 (0.000 to 0.002) |
| outcome_mortality | age_band = <50 | 1219 | 0.801 | 0.794 | 0.008 (0.000 to 0.015) |
| outcome_icd_aki | insurance_group = Other or missing | 266 | 0.887 | 0.884 | 0.003 (-0.004 to 0.010) |

## Tier 2: fidelity on measured slots (synthetic vs real train; reference = real test vs real train)

| Signal | Wasserstein / IQR | ref | MDD | ref | Hourly change W | ref | ACD | ref | Median count synth / real |
|---|---|---|---|---|---|---|---|---|---|
| heart_rate | 0.000 | 0.007 | 0.0000 | 0.0004 | 0.00 | 0.04 | 0.000 | 0.002 | 71 / 71 |
| sbp | 0.000 | 0.007 | 0.0000 | 0.0003 | 0.00 | 0.03 | 0.000 | 0.002 | 70 / 70 |
| dbp | 0.000 | 0.011 | 0.0000 | 0.0003 | 0.00 | 0.04 | 0.000 | 0.003 | 70 / 70 |
| resp_rate | 0.000 | 0.011 | 0.0000 | 0.0003 | 0.00 | 0.01 | 0.000 | 0.003 | 70 / 70 |
| spo2 | 0.000 | 0.006 | 0.0000 | 0.0003 | 0.00 | 0.01 | 0.000 | 0.003 | 70 / 70 |
| glucose | 0.000 | 0.028 | 0.0000 | 0.0010 | 0.00 | 16.49 | - | - | 5 / 5 |
| creatinine | 0.000 | 0.013 | 0.0000 | 0.0005 | 0.00 | 0.06 | - | - | 5 / 5 |
| sodium | 0.000 | 0.022 | 0.0000 | 0.0006 | 0.00 | 0.76 | - | - | 6 / 6 |

- Cross-signal correlation, mean absolute difference: synthetic 0.000, reference 0.008
- Discriminative score |accuracy - 0.5|: synthetic 0.089 (AUROC 0.357), reference 0.005 (AUROC 0.500)
- IP-alpha / IR-beta / recall at beta = 1 / authenticity: synthetic 0.996 / 0.411 / 1.000 / 0.240; reference 0.992 / 0.483 / 0.972 / 0.740 (read against the reference, not 1.0)
- DBP >= SBP share: synthetic 0.0000, real 0.0000
- Raw generator output outside the clip bounds: none

## Tier 3: clinical-event rates, hours 0-72 (equivalence margin +/- 0.05)

| Event | Real train | Synthetic | Gap (95% CI) | Reference gap | Equivalent |
|---|---|---|---|---|---|
| map_lt65_2h | 0.480 | 0.480 | 0.000 (-0.015 to 0.014) | 0.010 | yes |
| sbp_lt90_2h | 0.309 | 0.309 | 0.000 (-0.012 to 0.013) | -0.004 | yes |
| spo2_lt90_2h | 0.087 | 0.087 | 0.000 (-0.007 to 0.007) | -0.003 | yes |
| hr_gt100_2h | 0.604 | 0.604 | 0.000 (-0.014 to 0.014) | -0.011 | yes |
| hr_lt50_2h | 0.059 | 0.059 | 0.000 (-0.007 to 0.006) | -0.003 | yes |
| shock_index_ge1_2h | 0.448 | 0.448 | 0.000 (-0.013 to 0.013) | -0.003 | yes |
| aki_creatinine | 0.329 | 0.330 | 0.001 (-0.012 to 0.015) | 0.003 | yes |
| glucose_lt70_any | 0.067 | 0.067 | 0.000 (-0.007 to 0.007) | 0.005 | yes |
| glucose_lt54_any | 0.018 | 0.018 | 0.000 (-0.003 to 0.003) | 0.002 | yes |
| glucose_gt180_any | 0.392 | 0.392 | 0.000 (-0.014 to 0.014) | -0.013 | yes |
| sodium_lt135_any | 0.376 | 0.376 | 0.000 (-0.013 to 0.013) | 0.003 | yes |
| sodium_lt125_any | 0.024 | 0.024 | 0.000 (-0.004 to 0.004) | -0.000 | yes |
| sodium_gt145_any | 0.184 | 0.184 | 0.000 (-0.010 to 0.010) | -0.014 | yes |

Outcome gradient for mortality (event rate among deaths minus survivors): map_lt65_2h real 0.148 vs synth 0.148; sbp_lt90_2h real 0.123 vs synth 0.123; spo2_lt90_2h real 0.067 vs synth 0.067; hr_gt100_2h real 0.121 vs synth 0.121; hr_lt50_2h real -0.001 vs synth -0.001; shock_index_ge1_2h real 0.155 vs synth 0.155; aki_creatinine real 0.211 vs synth 0.213; glucose_lt70_any real 0.042 vs synth 0.042; glucose_lt54_any real 0.017 vs synth 0.017; glucose_gt180_any real 0.132 vs synth 0.132; sodium_lt135_any real 0.070 vs synth 0.070; sodium_lt125_any real 0.015 vs synth 0.015; sodium_gt145_any real 0.094 vs synth 0.094

## Privacy (not part of utility)

| Embedding | DCR median synth / holdout | Synth below holdout 5th pct | Exact-copy share | MIA AUROC | NNAA privacy loss |
|---|---|---|---|---|---|
| summary | 0.000 / 4.062 | 1.000 | 0.655 | 1.000 | 0.294 |
| raw_vitals | 0.000 / 15.266 | 1.000 | 0.517 | 1.000 | 0.265 |

Expected for a generator that does not memorize: share below the 5th percentile near 0.05, MIA AUROC near 0.5, privacy loss near 0.