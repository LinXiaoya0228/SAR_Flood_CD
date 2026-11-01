"""Day 6 smoke test: verifies label-free risk generation and probing isolation on Events 279 and 559."""

import sys
import json
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List
import torch
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.kuro_dataset import KuroUnlabeledValidationDataset
from src.models.snunet_wrapper import SNUNetWrapper
from src.tta.tent import TentAdapter
from src.risk.collector import RiskFeatureCollector
from src.risk.schema import validate_risk_feature_dict, check_forbidden_field_names


def compute_file_sha256(path: Path) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def run_smoke_test(cfg_path: Path, output_dir: Path, device: str = "cuda") -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    protocol_hash = compute_file_sha256(cfg_path)
    checkpoint_hash = cfg["checkpoint_sha256"]
    
    print("=" * 80)
    print("DAY 6 SMOKE TEST: RISK PROBING AND SCHEMA VALIDATION")
    print(f"Protocol:   {cfg_path.name} (SHA-256: {protocol_hash[:16]}...)")
    print(f"Checkpoint: {cfg['checkpoint_path']} (SHA-256: {checkpoint_hash[:16]}...)")
    print(f"Device:     {device}")
    print("=" * 80)

    # 1. Initialize unlabeled dataset for Events 279 and 559 only
    smoke_events = [279, 559]
    dataset = KuroUnlabeledValidationDataset(
        dataset_root=cfg["dataset_root"],
        manifest_path=cfg["manifest_path"],
        allowed_events=smoke_events,
        sealed_test_events=cfg["sealed_test_event_ids"],
        canonical_preevent=cfg["canonical_preevent"],
        preprocessing_version=cfg["preprocessing_version"],
    )

    # Group samples by event
    samples_by_event: Dict[int, List[Dict[str, Any]]] = {eid: [] for eid in smoke_events}
    for item in dataset.samples:
        samples_by_event[item["event_id"]].append(item)

    # 2. Load models
    source_model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    source_model.eval()

    candidate_model, _ = SNUNetWrapper.load_from_checkpoint(cfg["checkpoint_path"], device=device)
    adapter = TentAdapter(
        candidate_model,
        checkpoint_path=cfg["checkpoint_path"],
        learning_rate=cfg["learning_rate"],
        device=device,
    )

    collector = RiskFeatureCollector(
        source_model=source_model,
        protocol_hash=protocol_hash,
        checkpoint_hash=checkpoint_hash,
        device=device,
    )

    all_risk_records: List[Dict[str, Any]] = []
    all_predictions: Dict[str, torch.Tensor] = {}

    start_time = time.time()

    # 3. Stream 2 batches (32 samples) for each smoke event
    batch_size = cfg["batch_size"]
    samples_per_event = 32

    for eid in smoke_events:
        adapter.reset_to_source()
        event_items = samples_by_event[eid][:samples_per_event]
        print(f"\nProcessing smoke stream for Event {eid} ({len(event_items)} samples)...")

        for b_idx in range(0, len(event_items), batch_size):
            batch_slice = event_items[b_idx : b_idx + batch_size]
            actual_b_idx = (b_idx // batch_size) + 1

            # Load full sensory dict from dataset
            batch_tensors = [dataset[dataset.samples.index(item)] for item in batch_slice]

            sids = [t["sample_id"] for t in batch_tensors]
            xA = torch.stack([t["xA"] for t in batch_tensors]).to(device)
            xB = torch.stack([t["xB"] for t in batch_tensors]).to(device)
            pre1 = torch.stack([t["pre1"] for t in batch_tensors]).to(device)
            pre2 = torch.stack([t["pre2"] for t in batch_tensors]).to(device)
            post = torch.stack([t["post"] for t in batch_tensors]).to(device)
            sar_valid = torch.stack([t["valid_mask"] for t in batch_tensors]).to(device)

            # Step 1: Candidate Tent adaptation & prediction
            out = adapter.adapt_and_predict(xA, xB, sar_valid_mask=sar_valid)
            pred_class = out["predicted_class"].cpu()

            for j, sid in enumerate(sids):
                all_predictions[sid] = pred_class[j]

            # Step 2: Risk probing (guaranteed non-mutating)
            batch_sensory = {
                "sample_ids": sids,
                "xA": xA,
                "xB": xB,
                "pre1": pre1,
                "pre2": pre2,
                "post": post,
                "valid_mask": sar_valid,
            }

            risk_record = collector.collect_risk_record(
                candidate_adapter=adapter,
                batch_sensory=batch_sensory,
                run_id="smoke_tent_seed0",
                event_id=eid,
                seed=0,
                batch_idx=actual_b_idx,
            )

            all_risk_records.append(risk_record)
            print(
                f"  Batch {actual_b_idx}: Score={risk_record['predeclared_change_risk_score']:.4f} | "
                f"NullGrowth={risk_record['predeclared_signal_null_growth']:.4f} | "
                f"Erasure={risk_record['predeclared_signal_flood_erasure']:.4f} | "
                f"Entropy={risk_record['mean_prediction_entropy']:.4f}"
            )

    elapsed = time.time() - start_time

    # 4. Rigorous Smoke Verifications
    print("\n" + "=" * 80)
    print("VERIFYING SMOKE TEST INVARIANTS")
    print("=" * 80)

    # Verification A: Zero GT-derived field names in all records
    for i, rec in enumerate(all_risk_records):
        validate_risk_feature_dict(rec)
        forbidden = check_forbidden_field_names(set(rec.keys()))
        assert not forbidden, f"Forbidden GT fields found in record {i}: {forbidden}"
    print("[PASS] Check 1: Zero GT fields exist in risk records.")

    # Verification B: Key uniqueness
    record_keys = [(r["run_id"], r["event_id"], r["batch_idx"]) for r in all_risk_records]
    assert len(record_keys) == len(set(record_keys)), "Duplicate risk record keys detected!"
    assert len(all_predictions) == len(all_risk_records) * batch_size, "Prediction count mismatch!"
    print("[PASS] Check 2: All risk record keys and sample predictions are unique and aligned.")

    # Verification C: Zero sealed test event access
    accessed_eids = {r["event_id"] for r in all_risk_records}
    sealed_overlap = accessed_eids.intersection(set(cfg["sealed_test_event_ids"]))
    assert not sealed_overlap, f"CRITICAL: Sealed test events accessed: {sealed_overlap}!"
    print(f"[PASS] Check 3: Zero sealed test event overlap ({accessed_eids} accessed, 0 sealed).")

    # Verification D: All metrics finite
    for i, rec in enumerate(all_risk_records):
        for k, v in rec.items():
            if isinstance(v, (int, float)):
                assert np.isfinite(v), f"Non-finite value in record {i}, field {k}: {v}"
    print("[PASS] Check 4: All numeric risk features and scores are finite.")

    # Verification E: Protocol hash recorded accurately
    for rec in all_risk_records:
        assert rec["protocol_hash"] == protocol_hash, "Protocol hash mismatch in record!"
        assert rec["checkpoint_hash"] == checkpoint_hash, "Checkpoint hash mismatch in record!"
    print("[PASS] Check 5: Protocol hash and checkpoint hash recorded accurately.")

    # Save smoke artifacts
    smoke_risk_path = output_dir / "smoke_risk_features.json"
    with open(smoke_risk_path, "w", encoding="utf-8") as f:
        # Save records with list-serializable sample_ids
        json.dump(all_risk_records, f, indent=2)
    print(f"Saved smoke risk features to: {smoke_risk_path}")

    print("\nALL DAY 6 SMOKE CHECKS PASSED SUCCESSFULLY!")
    return {
        "status": "PASS",
        "protocol_hash": protocol_hash,
        "checkpoint_hash": checkpoint_hash,
        "records_generated": len(all_risk_records),
        "predictions_generated": len(all_predictions),
        "elapsed_seconds": elapsed,
    }


if __name__ == "__main__":
    cfg_file = PROJECT_ROOT / "configs/gate_b_risk_protocol.json"
    out_dir = PROJECT_ROOT / "outputs/day6_smoke"
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    res = run_smoke_test(cfg_file, out_dir, device=dev)
    print(json.dumps(res, indent=2))
