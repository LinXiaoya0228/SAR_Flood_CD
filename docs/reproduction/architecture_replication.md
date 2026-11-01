# Reproduction Guide: Architecture Transfer and Supplementary Replication

This document describes the second-backbone evaluation on SiamUnet_diff, the initial Seed 0 screening outcome, and the ongoing supplementary 3-seed replication study.

---

## 1. Background and Seed 0 Screening Result

To test whether the benefits of ChangeGuard (R2) are architecture-agnostic or specific to SNUNet-ECAM, the exact frozen recipes (R0 and R2) were applied to SiamUnet_diff across Folds A and B at Seed 0 (4 runs):

- **Grand Macro F1:**
  - R0 Baseline: $0.7238$
  - R2 Recipe: $0.7139$
  - Paired Delta: $\mathbf{-0.0099}$ (-0.99 pp)
- **Tail Reliability (Bottom-3 Tail F1):**
  - R0 Baseline: $0.4350$
  - R2 Recipe: $0.3902$
  - Paired Delta: $\mathbf{-0.0448}$ (-4.48 pp)
- **Decision:** Negative architecture transfer observed; triggered the pre-registered stopping rule.

---

## 2. Ongoing Supplementary 3-Seed Replication Study

To rigorously measure whether the Seed 0 degradation was an artifact of stochastic weight initialization or represents consistent cross-seed behavior, an authorized supplementary study extends SiamUnet_diff to Seeds 1 and 2 (8 additional training runs).

### Execution Status: **ONGOING**
- **Hardware:** Dedicated execution across 4 NVIDIA GPUs (GPUs 0, 1, 2, 3) in detached tmux sessions:
  - `cg_siam_gpu0`: Fold A, R0 (Seeds 1, 2)
  - `cg_siam_gpu1`: Fold A, R2 (Seeds 1, 2)
  - `cg_siam_gpu2`: Fold B, R0 (Seeds 1, 2)
  - `cg_siam_gpu3`: Fold B, R2 (Seeds 1, 2)
- **Protocol:** `docs/robust_source_training/rst8_siamunet_diff_supplementary_replication_protocol_v1.md`
- **Config Directory:** `configs/robust_source_training/rst8_siamunet_diff/`
- **Queue Launcher:** `scripts/robust_source_training/run_rst8_supp_gpu*.sh`

### Full 3-Seed Analysis Tool
Once all runs complete, the full 12-run matrix analysis is triggered via:
```bash
python tools/robust_source_training/run_rst8_full_3seed_analysis.py
```
This tool computes Fold A, Fold B, and Grand 18-event Macro F1, Bottom-3 Tail F1, seed variance ratios, and paired bootstrap 95% confidence intervals across all 3 seeds.

