# Synthetic MIMIC-IV utility report

Real train 10876 stays, real test 7338, synthetic 10984. AUROC margin 0.02, forecast margin 5% of TRTR MAE, event-rate margin +/- 0.05. Runtime 130.6 s.
Synthetic values were snapped to the real value grid before labels and metrics (resolution matching). Decision: non-inferior = upper CI bound of (TRTR - TSTR) below the margin; inferior = lower bound above it; otherwise inconclusive. 'Perfect-generator delta' trains on held-out real patients instead of synthetic data; UNDERPOWERED marks tasks where even that cannot pass.

## Tier 1: task utility (TSTR vs TRTR, real test set)

| Task | Prevalence real / synth | AUROC TRTR | AUROC TSTR | Delta (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| hypotension_h49_72 | 0.240 / 0.221 | 0.837 | 0.816 | 0.021 (0.015 to 0.027) | 0.020 | inconclusive | 0.005 (-0.002 to 0.011) |
| shock_index_h49_72 | 0.235 / 0.236 | 0.893 | 0.865 | 0.028 (0.022 to 0.034) | 0.020 | inferior | 0.002 (-0.003 to 0.007) |
| spo2_lt90_h49_72 | 0.036 / 0.022 | 0.850 | 0.829 | 0.021 (0.002 to 0.041) | 0.020 | inconclusive | 0.022 (-0.002 to 0.046) UNDERPOWERED |
| aki_new_h25_72 | 0.213 / 0.364 | 0.737 | 0.658 | 0.079 (0.066 to 0.096) | 0.020 | inferior | 0.005 (-0.007 to 0.018) |
| crossing_sbp | 0.057 / 0.057 | 0.887 | 0.870 | 0.018 (0.013 to 0.022) | 0.020 | inconclusive | -0.002 (-0.005 to 0.001) |
| crossing_heart_rate | 0.208 / 0.209 | 0.957 | 0.933 | 0.023 (0.021 to 0.026) | 0.020 | inferior | -0.000 (-0.001 to 0.001) |
| outcome_mortality | 0.178 / 0.180 | 0.763 | 0.750 | 0.012 (0.002 to 0.023) | 0.020 | inconclusive | 0.002 (-0.010 to 0.014) |
| outcome_icd_aki | 0.439 / 0.459 | 0.884 | 0.883 | 0.001 (-0.002 to 0.004) | 0.020 | non-inferior | -0.005 (-0.010 to -0.001) |

| Forecast task | MAE TRTR | MAE TSTR | MAE last value | Delta MAE (95% CI) | Margin | Decision | Perfect-generator delta (95% CI) |
|---|---|---|---|---|---|---|---|
| forecast_heart_rate | 5.26 | 6.29 | 5.56 | 1.03 (0.99 to 1.07) | 0.26 | inferior | -0.02 (-0.04 to -0.01) |
| forecast_sbp | 9.68 | 10.31 | 11.02 | 0.63 (0.59 to 0.67) | 0.48 | inferior | -0.02 (-0.04 to -0.01) |

Subgroup with the largest TRTR - TSTR delta for each task (all subgroups are in the JSON):

| Task | Subgroup | Test stays | AUROC TRTR | AUROC TSTR | Delta (95% CI) |
|---|---|---|---|---|---|
| hypotension_h49_72 | race_group = Hispanic/Latino | 240 | 0.841 | 0.812 | 0.029 (-0.008 to 0.065) |
| shock_index_h49_72 | insurance_group = Other or missing | 266 | 0.932 | 0.878 | 0.054 (0.025 to 0.091) |
| spo2_lt90_h49_72 | age_band = >=80 | 1271 | 0.881 | 0.840 | 0.040 (0.009 to 0.078) |
| aki_new_h25_72 | insurance_group = Medicaid | 875 | 0.774 | 0.678 | 0.096 (0.053 to 0.138) |
| crossing_sbp | race_group = Asian | 833 | 0.912 | 0.878 | 0.034 (0.007 to 0.065) |
| crossing_heart_rate | race_group = Hispanic/Latino | 933 | 0.959 | 0.927 | 0.032 (0.019 to 0.048) |
| outcome_mortality | race_group = Hispanic/Latino | 240 | 0.833 | 0.779 | 0.054 (0.012 to 0.109) |
| outcome_icd_aki | insurance_group = Other or missing | 266 | 0.887 | 0.879 | 0.008 (-0.007 to 0.022) |

## Tier 2: fidelity on measured slots (synthetic vs real train; reference = real test vs real train)

| Signal | Wasserstein / IQR | ref | MDD | ref | Hourly change W | ref | ACD | ref | Median count synth / real |
|---|---|---|---|---|---|---|---|---|---|
| heart_rate | 0.013 | 0.007 | 0.0005 | 0.0004 | 6.21 | 0.04 | 0.403 | 0.002 | 71 / 71 |
| sbp | 0.010 | 0.007 | 0.0003 | 0.0003 | 5.19 | 0.03 | 0.243 | 0.002 | 70 / 70 |
| dbp | 0.010 | 0.011 | 0.0003 | 0.0003 | 2.98 | 0.04 | 0.185 | 0.003 | 70 / 70 |
| resp_rate | 0.011 | 0.011 | 0.0003 | 0.0003 | 1.27 | 0.01 | 0.212 | 0.003 | 70 / 70 |
| spo2 | 0.003 | 0.006 | 0.0002 | 0.0003 | 0.79 | 0.01 | 0.203 | 0.003 | 70 / 70 |
| glucose | 0.016 | 0.028 | 0.0006 | 0.0010 | 41.62 | 16.49 | - | - | 5 / 5 |
| creatinine | 0.028 | 0.013 | 0.0007 | 0.0005 | 0.29 | 0.06 | - | - | 5 / 5 |
| sodium | 0.013 | 0.022 | 0.0006 | 0.0006 | 1.15 | 0.76 | - | - | 6 / 6 |

- Cross-signal correlation, mean absolute difference: synthetic 0.009, reference 0.008
- Discriminative score |accuracy - 0.5|: synthetic 0.076 (AUROC 0.599), reference 0.005 (AUROC 0.500)
- IP-alpha / IR-beta / recall at beta = 1 / authenticity: synthetic 0.995 / 0.494 / 0.964 / 0.741; reference 0.992 / 0.483 / 0.972 / 0.740 (read against the reference, not 1.0)
- DBP >= SBP share: synthetic 0.0000, real 0.0000
- Raw generator output outside the clip bounds: none

## Tier 3: clinical-event rates, hours 0-72 (equivalence margin +/- 0.05)

| Event | Real train | Synthetic | Gap (95% CI) | Reference gap | Equivalent |
|---|---|---|---|---|---|
| map_lt65_2h | 0.480 | 0.353 | -0.128 (-0.141 to -0.114) | 0.010 | NO |
| sbp_lt90_2h | 0.309 | 0.185 | -0.124 (-0.136 to -0.113) | -0.004 | NO |
| spo2_lt90_2h | 0.087 | 0.046 | -0.041 (-0.048 to -0.035) | -0.003 | yes |
| hr_gt100_2h | 0.604 | 0.465 | -0.139 (-0.151 to -0.125) | -0.011 | NO |
| hr_lt50_2h | 0.059 | 0.034 | -0.025 (-0.030 to -0.019) | -0.003 | yes |
| shock_index_ge1_2h | 0.448 | 0.325 | -0.123 (-0.135 to -0.110) | -0.003 | NO |
| aki_creatinine | 0.329 | 0.472 | 0.143 (0.131 to 0.157) | 0.003 | NO |
| glucose_lt70_any | 0.067 | 0.066 | -0.001 (-0.008 to 0.006) | 0.005 | yes |
| glucose_lt54_any | 0.018 | 0.019 | 0.000 (-0.003 to 0.004) | 0.002 | yes |
| glucose_gt180_any | 0.392 | 0.391 | -0.000 (-0.014 to 0.012) | -0.013 | yes |
| sodium_lt135_any | 0.376 | 0.366 | -0.009 (-0.023 to 0.004) | 0.003 | yes |
| sodium_lt125_any | 0.024 | 0.026 | 0.002 (-0.002 to 0.006) | -0.000 | yes |
| sodium_gt145_any | 0.184 | 0.180 | -0.004 (-0.015 to 0.007) | -0.014 | yes |

Outcome gradient for mortality (event rate among deaths minus survivors): map_lt65_2h real 0.148 vs synth 0.133; sbp_lt90_2h real 0.123 vs synth 0.102; spo2_lt90_2h real 0.067 vs synth 0.037; hr_gt100_2h real 0.121 vs synth 0.113; hr_lt50_2h real -0.001 vs synth -0.004; shock_index_ge1_2h real 0.155 vs synth 0.164; aki_creatinine real 0.211 vs synth 0.173; glucose_lt70_any real 0.042 vs synth 0.045; glucose_lt54_any real 0.017 vs synth 0.012; glucose_gt180_any real 0.132 vs synth 0.144; sodium_lt135_any real 0.070 vs synth 0.035; sodium_lt125_any real 0.015 vs synth 0.008; sodium_gt145_any real 0.094 vs synth 0.096

## Privacy (not part of utility)

| Embedding | DCR median synth / holdout | Synth below holdout 5th pct | Exact-copy share | MIA AUROC | NNAA privacy loss |
|---|---|---|---|---|---|
| summary | 4.052 / 4.062 | 0.048 | 0.000 | 0.493 | -0.016 |
| raw_vitals | 15.922 / 15.266 | 0.029 | 0.000 | 0.490 | 0.001 |

Expected for a generator that does not memorize: share below the 5th percentile near 0.05, MIA AUROC near 0.5, privacy loss near 0.