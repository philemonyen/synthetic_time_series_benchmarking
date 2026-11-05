# Synthetic MIMIC-IV utility report

Real train 10876 stays, real test 7338, synthetic 10984. AUROC margin 0.02, forecast margin 5% of TRTR MAE, event-rate margin +/- 0.05. Runtime 147.3 s.
Synthetic values were snapped to the real value grid before labels and metrics (resolution matching). Decision: non-inferior = upper CI bound of (TRTR - TSTR) below the margin; inferior = lower bound above it; otherwise inconclusive. 'Perfect-generator delta' trains on held-out real patients instead of synthetic data; UNDERPOWERED marks tasks where even that cannot pass.

## Tier 1: task utility (TSTR vs TRTR, real test set)

| Task | Prevalence real / synth | AUROC TRTR | AUROC TSTR | Delta (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| hypotension_h49_72 | 0.240 / 0.246 | 0.837 | 0.765 | 0.071 (0.063 to 0.081) | 0.020 | inferior | 0.005 (-0.002 to 0.011) |
| shock_index_h49_72 | 0.235 / 0.203 | 0.893 | 0.838 | 0.055 (0.048 to 0.062) | 0.020 | inferior | 0.002 (-0.003 to 0.007) |
| spo2_lt90_h49_72 | 0.036 / 0.034 | 0.850 | 0.805 | 0.045 (0.025 to 0.064) | 0.020 | inferior | 0.022 (-0.002 to 0.046) UNDERPOWERED |
| aki_new_h25_72 | 0.213 / 0.254 | 0.737 | 0.613 | 0.125 (0.107 to 0.143) | 0.020 | inferior | 0.005 (-0.007 to 0.018) |
| crossing_sbp | 0.057 / 0.072 | 0.887 | 0.837 | 0.050 (0.042 to 0.057) | 0.020 | inferior | -0.002 (-0.005 to 0.001) |
| crossing_heart_rate | 0.208 / 0.215 | 0.957 | 0.942 | 0.015 (0.013 to 0.017) | 0.020 | non-inferior | -0.000 (-0.001 to 0.001) |
| outcome_mortality | 0.178 / 0.176 | 0.763 | 0.620 | 0.142 (0.126 to 0.159) | 0.020 | inferior | 0.002 (-0.010 to 0.014) |
| outcome_icd_aki | 0.439 / 0.482 | 0.884 | 0.822 | 0.062 (0.054 to 0.069) | 0.020 | inferior | -0.005 (-0.010 to -0.001) |

| Forecast task | MAE TRTR | MAE TSTR | MAE last value | Delta MAE (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| forecast_heart_rate | 5.26 | 5.56 | 5.56 | 0.30 (0.28 to 0.32) | 0.26 | inferior | -0.02 (-0.04 to -0.01) |
| forecast_sbp | 9.68 | 10.08 | 11.02 | 0.40 (0.36 to 0.44) | 0.48 | non-inferior | -0.02 (-0.04 to -0.01) |

Subgroup with the largest TRTR - TSTR delta for each task (all subgroups are in the JSON):

| Task | Subgroup | Test stays | AUROC TRTR | AUROC TSTR | Delta (95% CI) |
|---|---|---|---|---|---|
| hypotension_h49_72 | insurance_group = Other or missing | 266 | 0.864 | 0.736 | 0.128 (0.074 to 0.180) |
| shock_index_h49_72 | insurance_group = Other or missing | 266 | 0.932 | 0.863 | 0.069 (0.034 to 0.105) |
| spo2_lt90_h49_72 | age_band = >=80 | 1271 | 0.881 | 0.797 | 0.083 (0.043 to 0.132) |
| aki_new_h25_72 | insurance_group = Other or missing | 231 | 0.767 | 0.594 | 0.174 (0.064 to 0.307) |
| crossing_sbp | age_band = >=80 | 4886 | 0.868 | 0.793 | 0.075 (0.054 to 0.094) |
| crossing_heart_rate | race_group = Asian | 842 | 0.949 | 0.926 | 0.023 (0.013 to 0.033) |
| outcome_mortality | insurance_group = Other or missing | 266 | 0.793 | 0.600 | 0.194 (0.126 to 0.267) |
| outcome_icd_aki | age_band = <50 | 1219 | 0.899 | 0.805 | 0.094 (0.075 to 0.111) |

## Tier 2: fidelity on measured slots (synthetic vs real train; reference = real test vs real train)

| Signal | Wasserstein / IQR | ref | MDD | ref | Hourly change W | ref | ACD | ref | Median count synth / real |
|---|---|---|---|---|---|---|---|---|---|
| heart_rate | 0.017 | 0.007 | 0.0013 | 0.0004 | 0.27 | 0.04 | 0.014 | 0.002 | 71 / 71 |
| sbp | 0.021 | 0.007 | 0.0011 | 0.0003 | 0.40 | 0.03 | 0.013 | 0.002 | 70 / 70 |
| dbp | 0.085 | 0.011 | 0.0018 | 0.0003 | 0.16 | 0.04 | 0.012 | 0.003 | 70 / 70 |
| resp_rate | 0.016 | 0.011 | 0.0012 | 0.0003 | 0.12 | 0.01 | 0.021 | 0.003 | 70 / 70 |
| spo2 | 0.017 | 0.006 | 0.0008 | 0.0003 | 0.07 | 0.01 | 0.017 | 0.003 | 70 / 70 |
| glucose | 0.105 | 0.028 | 0.0035 | 0.0010 | 47.50 | 16.49 | - | - | 5 / 5 |
| creatinine | 0.065 | 0.013 | 0.0020 | 0.0005 | 0.12 | 0.06 | - | - | 5 / 5 |
| sodium | 0.049 | 0.022 | 0.0025 | 0.0006 | 1.51 | 0.76 | - | - | 5 / 6 |

- Cross-signal correlation, mean absolute difference: synthetic 0.035, reference 0.008
- Discriminative score |accuracy - 0.5|: synthetic 0.500 (AUROC 1.000), reference 0.005 (AUROC 0.500)
- IP-alpha / IR-beta / recall at beta = 1 / authenticity: synthetic 0.970 / 0.181 / 0.173 / 0.733; reference 0.992 / 0.483 / 0.972 / 0.740 (read against the reference, not 1.0)
- DBP >= SBP share: synthetic 0.0000, real 0.0000
- Raw generator output outside the clip bounds: none

## Tier 3: clinical-event rates, hours 0-72 (equivalence margin +/- 0.05)

| Event | Real train | Synthetic | Gap (95% CI) | Reference gap | Equivalent |
|---|---|---|---|---|---|
| map_lt65_2h | 0.480 | 0.473 | -0.007 (-0.021 to 0.007) | 0.010 | yes |
| sbp_lt90_2h | 0.309 | 0.283 | -0.026 (-0.037 to -0.012) | -0.004 | yes |
| spo2_lt90_2h | 0.087 | 0.107 | 0.020 (0.012 to 0.026) | -0.003 | yes |
| hr_gt100_2h | 0.604 | 0.583 | -0.021 (-0.032 to -0.007) | -0.011 | yes |
| hr_lt50_2h | 0.059 | 0.053 | -0.006 (-0.012 to 0.000) | -0.003 | yes |
| shock_index_ge1_2h | 0.448 | 0.413 | -0.035 (-0.048 to -0.020) | -0.003 | yes |
| aki_creatinine | 0.329 | 0.333 | 0.005 (-0.007 to 0.018) | 0.003 | yes |
| glucose_lt70_any | 0.067 | 0.073 | 0.006 (0.000 to 0.014) | 0.005 | yes |
| glucose_lt54_any | 0.018 | 0.020 | 0.002 (-0.002 to 0.005) | 0.002 | yes |
| glucose_gt180_any | 0.392 | 0.343 | -0.048 (-0.062 to -0.036) | -0.013 | NO |
| sodium_lt135_any | 0.376 | 0.339 | -0.037 (-0.049 to -0.023) | 0.003 | yes |
| sodium_lt125_any | 0.024 | 0.023 | -0.001 (-0.005 to 0.004) | -0.000 | yes |
| sodium_gt145_any | 0.184 | 0.220 | 0.036 (0.025 to 0.048) | -0.014 | yes |

Outcome gradient for mortality (event rate among deaths minus survivors): map_lt65_2h real 0.148 vs synth 0.089; sbp_lt90_2h real 0.123 vs synth 0.045; spo2_lt90_2h real 0.067 vs synth -0.061; hr_gt100_2h real 0.121 vs synth 0.026; hr_lt50_2h real -0.001 vs synth 0.004; shock_index_ge1_2h real 0.155 vs synth 0.095; aki_creatinine real 0.211 vs synth 0.148; glucose_lt70_any real 0.042 vs synth 0.095; glucose_lt54_any real 0.017 vs synth -0.001; glucose_gt180_any real 0.132 vs synth 0.111; sodium_lt135_any real 0.070 vs synth -0.071; sodium_lt125_any real 0.015 vs synth -0.005; sodium_gt145_any real 0.094 vs synth 0.099

## Privacy (not part of utility)

| Embedding | DCR median synth / holdout | Synth below holdout 5th pct | Exact-copy share | MIA AUROC | NNAA privacy loss |
|---|---|---|---|---|---|
| summary | 4.078 / 4.062 | 0.057 | 0.000 | 0.502 | 0.001 |
| raw_vitals | 15.241 / 15.266 | 0.063 | 0.000 | 0.491 | 0.002 |

Expected for a generator that does not memorize: share below the 5th percentile near 0.05, MIA AUROC near 0.5, privacy loss near 0.