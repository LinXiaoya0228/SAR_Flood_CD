# Week 0 Pre-Change Repository Integrity State Record

- **Timestamp:** 2026-10-06T19:35:00+08:00
- **Project Root:** `/usr1/home/s126mdg22_06/FYP_SARCD`
- **Repository Directory:** `/usr1/home/s126mdg22_06/FYP_SARCD/ChangeGuard`
- **Host Name:** `gpu22`
- **Active Git Branch:** `main`
- **Current HEAD Commit:** `0d8dcec24c8c56609b4e318e317e66f5390c176b`
- **Commit Message:** `Checkpoint Days 8-10 pilot deliverables (protocols, tools, audits, final report)`

---

## 1. Initial Git Status

### 1.1 Working Tree Status
- **Staged files:** 0 (none)
- **Modified tracked files:** 0 (clean tracked state)
- **Untracked files:** Exactly 2 files present:
  1. `docs/MASTER_EXECUTION_PLAN.md` (SHA-256: `70fa605f4dadc09bfd98038bd164a08a6aaf36870c7578c34111dbdfef0b311f`)
  2. `docs/two_week_pilot_complete_evidence_report.md` (SHA-256: `a6d6fa5c38e4ba64591bde82e6fe9d6a2ee20d6c1da87de5a6cf15301875adc3`)

### 1.2 Full `git status --short` Output
```text
?? docs/MASTER_EXECUTION_PLAN.md
?? docs/two_week_pilot_complete_evidence_report.md
```

### 1.3 Repository Architecture & Nested Repositories Check
- **Nested repositories inside `ChangeGuard`:** None.
- **External sister repositories under `/usr1/home/s126mdg22_06/FYP_SARCD`:**
  - `third_party/KuroSiwo` (Git commit `4347ed173c4e48f5a9d578bb5fe8453706b08e5e`, clean tracked status)
  - `BRIGHT`
  - `UrbanSARFloods`
- **Unsaved or conflicting versions:** None. All tracked files are clean.

---

## 2. Integrity Certification on Scope Boundaries

- **Dataset / Checkpoint Exclusion Confirmation:** Confirmed that no raw dataset directory (`datasets/`, `data/`), model weights checkpoint (`checkpoints/`), or large generated artifact output directory (`outputs/`) was recursively hashed or staged into Git.
- **Sealed Test Boundary:** Confirmed that all 10 sealed test events (`[205, 277, 321, 411, 445, 561, 562, 1111002, 1111007, 1111013]`) remain strictly unaccessed.

---

## 3. Cryptographic Hashes of Tracked Files (61 Files)

### Root Files
- `.gitignore`: `76bee267a2be6996b6b7296550e1e516f59c479707d0320c566753b08114a575`
- `PROJECT_PROPOSAL.md`: `f8e0dd09035fcee41c467c30a3e3a844e65697aaad851b219d5b501264c470cf`

### Configuration Files (`configs/`)
- `configs/baseline_config.json`: `b465e19b3861422276c2d42eb100ab766b9c70d51b45693a79c7721167d88a0c`
- `configs/day8_tta_robustness_protocol.json`: `6aeeb4446da108ba5312409fff943cf10540e3ce51cd8dc3d19a013bdab44df7`
- `configs/day9_event_suitability_protocol.json`: `2a5893d6a5901f4e8f8ffa780eca7316254fa1ee542acc98c9aa9d0f07c20a4c`
- `configs/gate_b_risk_protocol.json`: `132ddb7ad11afc74dfd06485e92822516c7e820cce8f96a8c9c2e06a82b22074`
- `configs/kuro_event_split.json`: `fbf30fe0bab7b90c468aaf0fbdab7960a5cf76dd49dfb3ca51f1e01668fba85f`

### Documentation Files (`docs/`)
- `docs/checkpoint_provenance.md`: `8190bf2491dab226dcb7d56ab53888695c3aecaf9c9e572f11233e1e799be0fb`
- `docs/day2_baseline_audit.md`: `15be8a2c6d8490fd307e78d0d926a9fb59651c136714e54640f6ffd20b7770dc`
- `docs/day3_5_integrity_audit.md`: `43e3bfcb2831ccc54355ae921db082b1b9584e6063d0357283cc1e3342c7c243`
- `docs/day6_7_gate_b_audit.md`: `94edceefa5e5188e30e18c1b7259e5713243a6c9a8edda465de2dd96585d564b`
- `docs/day6_7_gate_b_validation.md`: `87610a07e50e07b6bc4f52ffa57c83969858f91ea5722a7d8a75a77f505da163`
- `docs/day8_10_progress.md`: `e62e3b521ff775eb5e38659a5a11327544b76373e145c6d0c0ab57cc20c8eb55`
- `docs/day8_robustness_audit.md`: `b144dc9520b4669cec8a50c1f5d3cbfe683bf912b2c0618d60d537fa95162e2d`
- `docs/day9_event_suitability_audit.md`: `d6e50a7c52794b37e492d32d1abc5ccc8f11e27a98cd2766a273cee522187be9`
- `docs/research_integrity_protocol.md`: `c81b12edea5b19747fc8177e68a46e19b356592e5fe17e2cc48183692eb947fb`
- `docs/two_week_pilot_final_report.md`: `0bcf595de775505dcaff0ce5f88b33556b6e3768769e3b6f70b248558f46cc47`

### Environment Specification Files (`environment/`)
- `environment/environment.yml`: `3ba90a5b43c4a88aef7e026e632a80f92a741c858e715c61a6483d7f9c7a8980`
- `environment/requirements-lock.txt`: `3e52c80c43166a342506c9f7293ea927045d3114b434808dde31a33cda2b8f93`

### Source Files (`src/`)
- `src/data/kuro_dataset.py`: `1b6569ded33d5131838ca5762c89f7572c732322ee7011808edf74b15d2718cb`
- `src/evaluation/completeness.py`: `ac8283c48bb59aaa89d997e3cd4c5966963d699ca036e2a4c9f55149bd566faa`
- `src/evaluation/metrics.py`: `29215ae5d7a1a8aff0ce4ba6e4f4524d9c2f3fdba27f4198bc9a1f75194e8a19`
- `src/evaluation/run_manifest.py`: `9f37d4e7b2bcd8ce05d9dcd8b1e27af9496aaede34e1b48738cd12bca3f68fec`
- `src/models/snunet_wrapper.py`: `818223a785bc1ea2e7e8984f060276d590cbc49b3360dea979a46abe7528ff88`
- `src/risk/__init__.py`: `caab638253c2299d87fa6c7f21205bed4af3125ca636d8a7d44a78642ce93637`
- `src/risk/collector.py`: `0d628b47520bc0f8f30d31bce4b8997b8ec7e809db805a60e18044af7fa772bd`
- `src/risk/indicators.py`: `a491adaed5ba9f3c3807499e336f03c3bc27332c51f8d4234ccb67176562e32b`
- `src/risk/probes.py`: `c7e953dc5ba32c0d67d188942985b4e45c5a0c25c4ddb5b47a4536eced1dc8d0`
- `src/risk/schema.py`: `ba8d36b002ed8d8139ac91338238b2218e2d1adbaece9a0f01040d0a8dccd21a`
- `src/risk/scores.py`: `a64c481249f8fa6f19ce8dc3da3558f4f1351f601cec87a69a6875a369de3d80`
- `src/tta/adabn.py`: `e71336b4a21b0a84f3010b596e4320b603d667e6ab3574f157d222a4512ce1e6`
- `src/tta/base.py`: `16fd87557d8c98a27a3825d9825725a3fccabfbdfb1043cc1ac04081393db462`
- `src/tta/entropy.py`: `5aa231a398ef7d693c3c5acdf1c871fc46a8633ed967f351e91d5121ad0a53ad`
- `src/tta/source_only.py`: `403f2f4b2e02f86390866c395f9093600891c2b0f31eaa3092d8ba626cfea317`
- `src/tta/tent.py`: `d1db5b3fa376078ae5dba9b302af770a8636f8a0caf59ddbeb486e3a75a1d684`

### Test Suite Files (`tests/`)
- `tests/test_completeness.py`: `c4f27f90793e4d264a3303d53ad0cd312f9f2df637eb98a41283eb1a19275561`
- `tests/test_metrics.py`: `0d140b63724538a945aef3710ace16e605e66530143c7f55e600507777ee0388`
- `tests/test_preprocessing.py`: `2b107b815c80ece232c6c3914299d95f9fadb4c045b8e5f328043245e2e74fb8`
- `tests/test_risk_indicators.py`: `349356a4843f4609508c5fcd06e3bc94f8aa19302abb3e4a50ce0136e3664353`
- `tests/test_risk_probing_isolation.py`: `46f8a14f35304c6d6983cfd78fe7a6a4088d5e98b3b1819101a19d445ee16acb`
- `tests/test_risk_schema.py`: `59fdd3e452c6489b7bb563f486ca5ae89654f6509d39da7b2450b400be8ea105`
- `tests/test_risk_thresholds.py`: `af99332a0e4f335ea8c5478105edd53ee646c72d8bc8a9f4fcbac52255dec842`
- `tests/test_run_manifest.py`: `8a944f396e5d26435fd2d25d0316bd72028f652869c67faf69868bcc7f0e5ca5`
- `tests/test_tta_isolation.py`: `d907544c93472f6a6fdb6f57f64fd72291f97bb0a76f297ef5a321dc4c22313e`

### Execution and Diagnostic Tool Scripts (`tools/`)
- `tools/audit_day2_deep.py`: `197b348e23d9e304143833b5ee106f05c6c5d8a3486916ded135b7b6e0a8651e`
- `tools/audit_kuro_simple.py`: `c82ba790b54c66fc696fddb52f4f8f3ed485a43c4118db4aa10b3efbacbcd414`
- `tools/audit_kuro_test_manifest.py`: `bd66ca09159e08744bfdc788e76b8da88d97e46382b2cf4dedd787c21b23c1a5`
- `tools/eval_val_baseline.py`: `0d255967400c420d2690ccde694f383ccd91b15a9ff77314ebc7262a62992d63`
- `tools/eval_val_baseline_corrected.py`: `777c258a2ee901590c6d89f12331c691e1f0e9f19ae8c016940fc0f812e527c4`
- `tools/finish_day1.py`: `750e32c15183e1d95cabf056f970160796ec048f55f85d7cc218866c424d949e`
- `tools/run_day2_supplemental_audit.py`: `f2515d828de863ece7ebaa6bb7d4fc111c5160c3344e7fb083dd1886c2a174b1`
- `tools/run_day3_smoke_tests.py`: `d2c3697ca79360d44619aee4aa377ab7a4b0e95f544aac2ad36fc42239c6c997`
- `tools/run_day6_smoke_tests.py`: `1ad4a3205c994b2622ed3ab574a49c44bc18152c773493d01570f9fc516cdbe3`
- `tools/run_day8_robustness.py`: `871d49436ad7ae1f8b024945461674b179cfbc09d9622ecc750f08cbef055b4c`
- `tools/run_day9_event_suitability.py`: `355d929670b662a9f4dcb9a4bdc151b64d8f5d6a42e6ff889b8a3c1f0e27ce2a`
- `tools/run_full_evaluation_comparison.py`: `302ed9c9a102325a9ff73820c9d920499ed56c1d8dcc9573ae7c64e5de586e09`
- `tools/run_gate_b_independent_validation.py`: `400a4f3c364a0f64b19d1bcd377c85306f08a6e1fc22a55defdb6f0e4b0c533c`
- `tools/run_gate_b_offline_evaluation.py`: `4b23dd2a2ca351d00ceea2f4f6472b379515b19e977d1631f009ce8fc260fed4`
- `tools/run_gate_b_risk_generation.py`: `866afcc03b08bf4599121abec66ffbef27ab33482990688b21b24479e603f28b`
- `tools/run_tta_evaluation.py`: `cb72be372b39ad6fd99b9a9d3bcbfcd4bce099c08b1d5dcd16ec2f3fb8a21b8e`
- `tools/smoke_test_real_sample.py`: `493df5e5878c5b57a2267a171ce1f7855f7f829128c6f9e01ce85c86cdabb452`
