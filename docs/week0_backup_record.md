# Week 0 Recoverable Code-Level Backup Record

- **Timestamp:** 2026-10-06T19:31:20+08:00
- **Host / Device:** `gpu22`
- **Backup Directory:** `/usr1/home/s126mdg22_06/FYP_SARCD/backups`
- **Source Repository:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
- **Active Commit:** `0d8dcec24c8c56609b4e318e317e66f5390c176b` (branch `main`)

---

## 1. Backup Artifacts Summary

| Artifact Type | File Path | File Size | SHA-256 Digest | Verification Method & Result |
| :--- | :--- | :---: | :--- | :--- |
| **Git Bundle** | `/usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_refs_20261006_193120.bundle` | 199 KB (203,728 bytes) | `ccb68f97461a81108bc081b0b79bb6f789ff4ccd40e5e748a9991bc7742598cb` | `git bundle verify` returned: `The bundle records a complete history. ... is okay` |
| **Working-Tree Archive** | `/usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_worktree_20261006_193120.tar.gz` | 192 KB (196,176 bytes) | `e1d57d80a7da2364bc936002faa60aee30297083e1622fa9c84fddcb1e4e1f1a` | `tar -tzvf` successfully listed all 61 tracked files plus untracked docs |

---

## 2. Exclusions and Preservation Policies

1. **Strictly Excluded Paths:**
   - Raw dataset files (`datasets/`, `data/`)
   - Pre-trained checkpoints and weight tensors (`checkpoints/`, `*.pt`, `*.pth`)
   - Generated metric outputs and run logs (`outputs/`, `*.log`, `*.csv`, `*.json` in outputs)
   - Cache and bytecode files (`__pycache__/`, `*.pyc`, `.pytest_cache/`)
   - Virtual environments (`miniconda3/`, `.venv/`)
   - Internal Git database (`.git/`)
2. **Preserved Working-Tree Content:**
   - All source modules (`src/`)
   - All configuration protocols (`configs/`)
   - All test suites (`tests/`)
   - All execution and diagnostic tools (`tools/`)
   - Environment definitions (`environment/`)
   - Project proposals and documentation (`docs/`, `PROJECT_PROPOSAL.md`, `.gitignore`)
   - Preserved untracked documents created prior to backup:
     - `docs/MASTER_EXECUTION_PLAN.md`
     - `docs/two_week_pilot_complete_evidence_report.md`

---

## 3. Recovery Verification Instructions

To restore from this backup in case of emergency:

```bash
# To inspect or restore git history:
git clone /usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_refs_20261006_193120.bundle restored_changeguard

# To restore code-level working tree:
tar -xzvf /usr1/home/s126mdg22_06/FYP_SARCD/backups/changeguard_worktree_20261006_193120.tar.gz -C <target_directory>
```

Neither backup file is tracked or committed into the repository.
