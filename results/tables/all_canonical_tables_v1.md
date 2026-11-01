# ChangeGuard RST10: Canonical Tables Package

**Generated via deterministic reconstruction pipeline.**

---

## Table 1: Table1 Component Staircase

*Source CSV: `results/tables/table1_component_staircase.csv`*

| recipe_id | components | macro_f1 | precision | recall | iou | worst_event_f1 | bottom3_mean_f1 | delta_f1_vs_r0 | delta_bottom3_vs_r0 | evidence_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 | None (Matched Baseline) | 0.763134 | 0.771665 | 0.779724 | 0.644164 | 0.419705 | 0.562984 | 0.0 | 0.0 | Discovery (Seed 0, Group C, 9 events) |
| R1 | M1 (Event Balance) | 0.756085 | 0.820078 | 0.730677 | 0.639089 | 0.325309 | 0.542608 | -0.007049 | -0.020375 | Discovery (Seed 0, Group C, 9 events) |
| R2 | M1 (Event Balance) + M2 (Class Weights) | 0.765146 | 0.830619 | 0.731456 | 0.643778 | 0.413698 | 0.586891 | 0.002012 | 0.023907 | Discovery (Seed 0, Group C, 9 events) |
| R3 | M1 (Event Balance) + M2 (Class Weights) + M3 (SAR Aug) | 0.769657 | 0.81913 | 0.739951 | 0.647546 | 0.481716 | 0.58243 | 0.006523 | 0.019446 | Discovery (Seed 0, Group C, 9 events) |

---

## Table 2: Table2 Primary Source Confirmation

*Source CSV: `results/tables/table2_primary_source_confirmation.csv`*

| partition | seed | r0_macro_f1 | r2_macro_f1 | delta_f1 | r0_precision | r2_precision | r0_recall | r2_recall | r0_iou | r2_iou |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fold A (9 held-out events) | Seed 0 | 0.733 | 0.7565 | 0.0235 | 0.7599 | 0.736 | 0.7495 | 0.8212 | 0.6084 | 0.6276 |
| Fold A (9 held-out events) | Seed 1 | 0.7484 | 0.7636 | 0.0152 | 0.8091 | 0.7483 | 0.738 | 0.8101 | 0.6196 | 0.6304 |
| Fold A (9 held-out events) | Seed 2 | 0.7566 | 0.7427 | -0.0139 | 0.725 | 0.7044 | 0.8287 | 0.8306 | 0.6249 | 0.6081 |
| Fold A Summary | Mean ± SD | 0.7460 ± 0.0120 | 0.7543 ± 0.0106 | +0.0083 | 0.7647 | 0.7296 | 0.7721 | 0.8206 | 0.6176 | 0.6220 |
| Fold B (9 held-out events) | Seed 0 | 0.6937 | 0.7159 | 0.0222 | 0.7584 | 0.7629 | 0.7101 | 0.7309 | 0.5613 | 0.5856 |
| Fold B (9 held-out events) | Seed 1 | 0.6982 | 0.7283 | 0.0302 | 0.7816 | 0.7926 | 0.694 | 0.7172 | 0.5652 | 0.5975 |
| Fold B (9 held-out events) | Seed 2 | 0.6912 | 0.6956 | 0.0044 | 0.7718 | 0.718 | 0.6899 | 0.7381 | 0.5567 | 0.5579 |
| Fold B Summary | Mean ± SD | 0.6943 ± 0.0035 | 0.7133 ± 0.0165 | +0.0189 | 0.7706 | 0.7578 | 0.6980 | 0.7287 | 0.5610 | 0.5803 |
| Combined Grand Matrix (18 unique events) | All Seeds (Grand) | 0.7202 | 0.7338 | +0.0136 (95% CI: [-0.0045, +0.0351]) | 0.7676 | 0.7437 | 0.7350 | 0.7747 | 0.5893 | 0.6012 |
| Tail Reliability: Bottom-3 Mean F1 | All Seeds (Tail) | 0.3987 | 0.4498 | +0.0511 | — | — | — | — | — | — |
| Tail Reliability: Worst Event F1 | All Seeds (Worst) | 0.2991 | 0.2929 | -0.0062 | — | — | — | — | — | — |

---

## Table 3: Table3 Architecture Transfer

*Source CSV: `results/tables/table3_architecture_transfer.csv`*

| partition | seed | r0_f1 | r2_f1 | delta_f1 | r0_precision | r2_precision | r0_recall | r2_recall | r0_bottom3_f1 | r2_bottom3_f1 | delta_bottom3 | r0_worst_f1 | r2_worst_f1 | delta_worst | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Fold A (9 held-out events) | Seed 0 | 0.7355 | 0.7134 | -0.0221 | 0.8777 | 0.7996 | 0.6672 | 0.7079 | — | — | — | — | — | — | Transfer not supported (Degraded) |
| Fold B (9 held-out events) | Seed 0 | 0.7121 | 0.7145 | 0.0024 | 0.8105 | 0.8283 | 0.6842 | 0.6664 | — | — | — | — | — | — | Marginal / Neutral (+0.0024) |
| Combined Overall (18 events) | Seed 0 | 0.7238 | 0.7139 | -0.0099 | 0.8441 | 0.814 | 0.6757 | 0.6872 | 0.435 | 0.3902 | -0.0448 | 0.3565 | 0.266 | -0.0905 | STOP AT SEED 0 (P1 Fail, P2 Fail, CI: [-0.0478, +0.0229]) |

---

## Table 4: Table4 Deployable Checkpoint Ledger

*Source CSV: `results/tables/table4_deployable_checkpoint_ledger.csv`*

| seed | run_id | architecture | recipe | source_events | epochs | optimizer_steps | runtime_seconds | runtime_hours | peak_vram_mb | checkpoint_sha256 | config_sha256 | terminal_status | exit_code | weights_reload_verified | parameters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | rst9_r2_snunet_all27_s0_v1 | SNUNet_ECAM | R2 | 27 | 16 | 10736 | 11682.69 | 3.25 | 4913.16 | 9611df27a18cd1cb16e60cb74e10a06df7afe4947d948bb3c0e94ef39a10dfb7 | 67b83783cf2b4a8b2d728a7cfc4416a93e2b88e5acad9ff8b1eaa6aa48ecec5b | COMPLETED_SCIENTIFIC_RUN | 0 | True | 12034819 |
| 1 | rst9_r2_snunet_all27_s1_v1 | SNUNet_ECAM | R2 | 27 | 16 | 10736 | 12247.64 | 3.4 | 4913.16 | 279dc73c8f6d0274031a33f3bd58a7af632dc2248c709db4c6a66404b21efbd2 | 9ef0ef9a506ba4e71bdc005d1a8d8e5d61a2c90fe8a66aeca7e0a7c45334db7c | COMPLETED_SCIENTIFIC_RUN | 0 | True | 12034819 |
| 2 | rst9_r2_snunet_all27_s2_v1 | SNUNet_ECAM | R2 | 27 | 16 | 10736 | 11564.47 | 3.21 | 4913.16 | b057a01820fdeaa596d8340eb550a174a2b8ca8459db0ca66dc08a4aaf929893 | 1f0255fa3f6c790131cac2a9a8d5c66a8afcef9f2e7d73bcf25882b7d9bd2696 | COMPLETED_SCIENTIFIC_RUN | 0 | True | 12034819 |

---

## Table 5: Table5 Historical Target Extension

*Source CSV: `results/tables/table5_historical_target_extension.csv`*

| partition | seed | r0_f1 | r2_f1 | delta_f1 | r0_dev_f1 | r2_dev_f1 | r0_test_f1 | r2_test_f1 | r0_bottom3_f1 | r2_bottom3_f1 | historical_disclaimer |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| All 16 Historical Events | Seed 0 | 0.7009 | 0.7355 | 0.0346 | 0.5949 | 0.611 | 0.7645 | 0.8102 | 0.4354 | 0.4426 | Observed historically in Weeks 3-5; retrospective extension only |
| All 16 Historical Events | Seed 1 | 0.6616 | 0.7337 | 0.0721 | 0.5884 | 0.6218 | 0.7055 | 0.8009 | 0.3191 | 0.4629 | Observed historically in Weeks 3-5; retrospective extension only |
| All 16 Historical Events | Seed 2 | 0.6994 | 0.7447 | 0.0453 | 0.5885 | 0.6279 | 0.7659 | 0.8149 | 0.412 | 0.4767 | Observed historically in Weeks 3-5; retrospective extension only |
| All 16 Historical Events | Mean ± SD | 0.6873 ± 0.0223 | 0.7380 ± 0.0059 | +0.0507 (95% CI: [+0.0234, +0.0845]) | 0.5906 | 0.6202 (+0.0296) | 0.7453 | 0.8086 (+0.0633) | 0.3889 | 0.4607 (+0.0719) | Observed historically in Weeks 3-5; retrospective extension only |

---

## Table 6: Table6 Runtime And Resource Cost

*Source CSV: `results/tables/table6_runtime_and_resource_cost.csv`*

| model_architecture | method_recipe | parameter_count | training_optimizer_steps | training_runtime_hours | peak_vram_mb | inference_throughput_samples_sec | target_time_adaptation_overhead | checkpoint_size_mb | operational_notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SNUNet_ECAM | R0 (Baseline Standard CE) | 12034819 | 10736 | 3.15h (avg) | 4913.16 | ~64 samples/s | 0.0s (Zero target-time overhead) | 46.1 | Standard pure source model; baseline inference |
| SNUNet_ECAM | R2 (Event-Bal + Class Weights) | 12034819 | 10736 | 3.29h (avg) | 4913.16 | ~64 samples/s | 0.0s (Zero target-time overhead) | 46.1 | Identical inference graph and runtime cost as R0; zero test adaptation |
| SiamUnet_diff | R0 (Baseline Standard CE) | 1335011 | 2688 | 0.58h | 3124.0 | ~180 samples/s | 0.0s (Zero target-time overhead) | 5.4 | Lightweight Siamese backbone; 5 epochs budget |
| SiamUnet_diff | R2 (Event-Bal + Class Weights) | 1335011 | 2688 | 0.59h | 3124.0 | ~180 samples/s | 0.0s (Zero target-time overhead) | 5.4 | Transfer not supported; stopped at Stage A |
| SNUNet_ECAM | AdaBN / BN-no-update (TTA) | 12034819 | N/A (Uses R0 ckpt) | 0.0h | 4913.16 | ~58 samples/s | Low (Online batch re-estimation forward pass) | 46.1 | Requires test batch re-estimation; state mutations |
| SNUNet_ECAM | Tent (Entropy Minimization TTA) | 12034819 | N/A (Uses R0 ckpt) | 0.0h | 5840.0 | ~22 samples/s | High (Online test backprop on BN affine parameters) | 46.1 | Requires gradient backprop on live test data; risk of collapse |
| SNUNet_ECAM | CoTTA (Continual TTA) | 12034819 | N/A (Uses R0 ckpt) | 0.0h | 8950.0 | ~11 samples/s | Very High (Teacher-student EMA + aug consistency + reset cache) | 92.2 | Extreme test memory footprint and 6x slower inference |

---

## Table 7: Table7 Contextual Tta Comparison

*Source CSV: `results/tables/table7_contextual_tta_comparison.csv`*

| method | category | target_events | adaptation_mode | target_time_updates | macro_f1 | sd_f1 | delta_vs_source | ci_95 | ci_crosses_zero | compatibility_limitations |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R0 Source-Only Baseline | Source Training Anchor | 10 Former Test Events | None (Frozen Source) | Zero | 0.7453 | 0.0345 | 0.0 | N/A | N/A | Directly matched benchmark anchor for all 10 former test evaluations |
| bn_no_update (AdaBN) | Test-Time Adaptation | 10 Former Test Events | Online batch stats re-estimation | Running BN mean/var | 0.7599 | 0.0158 | 0.0146 | [-0.0148, +0.0560] | Yes (p=0.45) | Contextual: requires batch size >=16 at test time; non-deterministic batch ordering |
| Tent | Test-Time Adaptation | 10 Former Test Events | Entropy minimization on BN scale/shift | Online gradient descent | 0.7628 | 0.0147 | 0.0175 | [-0.0119, +0.0586] | Yes (p=0.33) | Contextual: online backprop; predictions depend on arrival stream order |
| CoTTA | Test-Time Adaptation | 10 Former Test Events | Teacher-student EMA + stochastic restoration | Online teacher/student updates | 0.7669 | 0.0132 | 0.0216 | [-0.0077, +0.0634] | Yes (p=0.21) | Contextual: heavy dual-model state in VRAM; non-deterministic streaming state |
| Recipe R2 (RST9 Deployed) | Robust Source Training | 10 Former Test Events | Frozen Deployed Model (M1+M2) | Zero (Pure Source Inference) | 0.8086 | 0.0072 | 0.0633 | [+0.0312, +0.0984] | No (p<0.001) | Retrospective historical-target extension; pure feed-forward source inference |
| EATA (Fisher + Entropy Filter) | Test-Time Adaptation | 6 Development Events | Fisher-regularized entropy filtering | Selective BN updates | 0.5794 | 0.0162 | -0.0112 | N/A | Harmful | Evaluated on 6 dev events in Week 9; negative delta (-0.0112) vs source baseline 0.5906 |
| SAR (Sharpness-Aware TTA) | Test-Time Adaptation | 6 Development Events | Sharpness-aware minimization + margin filtering | Selective gradient updates | 0.5806 | 0.0168 | -0.01 | N/A | Harmful | Evaluated on 6 dev events in Week 9; negative delta (-0.0100) vs source baseline 0.5906 |

---

