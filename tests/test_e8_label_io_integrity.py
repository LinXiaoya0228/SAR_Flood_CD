#!/usr/bin/env python3
"""Lifecycle regression tests for the E8 v2 physical label-I/O repair."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import torch

from tools import run_e8_confirmation_matrix as e8mod
from tools.run_e8_confirmation_matrix import E8ConfirmationRunner


REPO_ROOT = Path(__file__).resolve().parents[1]


class SpyDataset:
    def __init__(self, records, load_labels, trace):
        self.samples = [
            {"sample_id": r["sample_id"], "event_id": r["event_id"], "relative_path": "synthetic"}
            for r in records
        ]
        self.records = records
        self.load_labels = load_labels
        self.trace = trace
        self.index_count = 0

    def __getitem__(self, index):
        self.index_count += 1
        rec = self.records[index]
        self.trace.append(("offline_index" if self.load_labels else "sensory_index", rec["sample_id"]))
        result = {
            "sample_id": rec["sample_id"],
            "event_id": rec["event_id"],
            "xA": rec["xA"],
            "xB": rec["xB"],
            "valid_mask": rec["valid_mask"],
        }
        if self.load_labels:
            result["label"] = rec["label"]
        return result


class SpyAdapter:
    mode = "reset_per_event"

    def __init__(self, trace, manifest_path):
        self.trace = trace
        self.manifest_path = manifest_path
        self.calls = 0
        self.num_updates_applied = 0
        self.num_updates_skipped = 0
        self.num_updates_failed = 0
        self.first_manifest_hash = None

    def reset_to_source(self):
        self.trace.append(("reset", None))

    def adapt_and_predict_step(self, *, xA, xB, valid_mask, sample_ids, event_id, **kwargs):
        self.calls += 1
        self.trace.append(("adapter", tuple(sample_ids)))
        assert self.manifest_path.is_file()
        self.first_manifest_hash = hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
        assert not kwargs
        assert torch.equal(valid_mask, torch.ones_like(valid_mask))
        batch, _, height, width = xA.shape
        pred = torch.zeros((batch, height, width), dtype=torch.long)
        probs = torch.zeros((batch, 3, height, width), dtype=torch.float32)
        probs[:, 0] = 1.0
        self.num_updates_applied += 1
        return {
            "primary_prediction": {"pred_cand_pre": pred, "probs_cand_pre": probs},
            "adaptation_metadata": {
                "step_count": self.calls,
                "action": "update",
                "update_applied": True,
                "selected_pixels": int(valid_mask.sum()),
            },
            "risk_evidence": {"mean_entropy": 0.0, "mean_max_confidence": 1.0},
        }


class TestE8LabelIOIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.tmp = Path(self.temp.name)
        self.trace = []
        self.records = []
        for i in range(2):
            self.records.append({
                "sample_id": f"sample_{i}",
                "event_id": 520,
                "xA": torch.zeros((2, 4, 4), dtype=torch.float32),
                "xB": torch.ones((2, 4, 4), dtype=torch.float32),
                "valid_mask": torch.ones((4, 4), dtype=torch.uint8),
                "label": torch.zeros((4, 4), dtype=torch.long),
            })

        base_config = json.loads((REPO_ROOT / "configs/changeguard/e8_confirmation_matrix_v2.json").read_text())
        checkpoint = self.tmp / "checkpoint.pt"
        checkpoint.write_bytes(b"synthetic checkpoint")
        base_config["source_checkpoints"]["seeds"]["0"]["checkpoint_path"] = str(checkpoint)
        base_config["source_checkpoints"]["seeds"]["0"]["expected_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        self.config_path = self.tmp / "config.json"
        self.config_path.write_text(json.dumps(base_config), encoding="utf-8")

        self.ledger_path = self.tmp / "ledger.json"
        self.ledger_path.write_text(json.dumps({
            "reused_cells": [],
            "execute_new_cells": [{
                "stream_id": "synthetic_stream",
                "method": "cotta_reset",
                "seed": 0,
                "order": "order_a",
                "reset_mode": "reset_per_event",
                "events": [520],
                "status": "EXECUTE_NEW",
            }],
        }), encoding="utf-8")
        self.output_root = self.tmp / "output"
        self.runner = E8ConfirmationRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            output_dir=self.output_root,
            device="cpu",
        )
        self.adapt_ds = SpyDataset(copy.deepcopy(self.records), load_labels=False, trace=self.trace)
        self.offline_ds = SpyDataset(copy.deepcopy(self.records), load_labels=True, trace=self.trace)
        self.runner._adapt_dataset = self.adapt_ds
        self.runner._offline_dataset = self.offline_ds
        self.adapter = SpyAdapter(self.trace, self.output_root / "synthetic_stream/initial_manifest.json")
        self.runner.instantiate_model = mock.Mock(return_value=torch.nn.Identity())
        self.runner.instantiate_adapter = mock.Mock(return_value=self.adapter)

    def tearDown(self):
        self.temp.cleanup()

    def execute(self):
        return self.runner.execute_stream(
            stream_id="synthetic_stream",
            method_name="cotta_reset",
            seed=0,
            order_id="order_a",
            reset_mode="reset_per_event",
            event_subset=[520],
        )

    def test_deferred_label_read_occurs_after_atomic_persistence(self):
        original = e8mod.atomic_write_npz

        def traced_write(path, **arrays):
            original(path, **arrays)
            self.trace.append(("persisted", tuple(str(x) for x in arrays["sample_ids"])))

        with mock.patch.object(e8mod, "atomic_write_npz", side_effect=traced_write):
            summary = self.execute()

        persist_pos = next(i for i, x in enumerate(self.trace) if x[0] == "persisted")
        offline_pos = next(i for i, x in enumerate(self.trace) if x[0] == "offline_index")
        adapter_pos = next(i for i, x in enumerate(self.trace) if x[0] == "adapter")
        self.assertLess(adapter_pos, persist_pos)
        self.assertLess(persist_pos, offline_pos)
        self.assertEqual(self.offline_ds.index_count, 2)
        self.assertEqual(summary["deferred_gt_reads"], 2)

        audits = list((self.output_root / "synthetic_stream/event_520/label_io_audits").glob("*.json"))
        self.assertEqual(len(audits), 1)
        audit = json.loads(audits[0].read_text())
        self.assertTrue(audit["label_read_after_persistence"])
        self.assertEqual(audit["sample_ids"], ["sample_0", "sample_1"])

    def test_persistence_failure_reads_zero_labels_and_records_failure(self):
        with mock.patch.object(e8mod, "atomic_write_npz", side_effect=OSError("injected persistence failure")):
            with self.assertRaisesRegex(OSError, "injected persistence failure"):
                self.execute()
        self.assertEqual(self.offline_ds.index_count, 0)
        stream_dir = self.output_root / "synthetic_stream"
        self.assertTrue((stream_dir / "failure.json").is_file())
        status = json.loads((stream_dir / "completion_status.json").read_text())
        self.assertEqual(status["status"], "FAILED")
        self.assertFalse(any(stream_dir.glob("event_*/label_io_audits/*.json")))

    def test_manifest_precedes_adapter_is_complete_and_immutable(self):
        self.execute()
        manifest_path = self.output_root / "synthetic_stream/initial_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        required = {
            "run_id", "stream_id", "git_head", "git_tracked_dirty", "config_sha256",
            "ledger_sha256", "runner_sha256", "adapter_sha256", "checkpoint_path",
            "checkpoint_sha256", "split_sha256", "dataset_manifest_sha256", "event_ids",
            "expected_sample_counts", "expected_batch_counts", "prediction_timing", "reset_mode",
            "order", "device", "sealed_test_rejection_status", "label_io_policy", "start_timestamp_utc",
        }
        self.assertTrue(required.issubset(manifest))
        after_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        self.assertEqual(self.adapter.first_manifest_hash, after_hash)
        final = json.loads((self.output_root / "synthetic_stream/manifest.json").read_text())
        self.assertEqual(final["initial_manifest_sha256"], after_hash)

    def test_completed_resumption_is_idempotent_and_partial_resume_refuses(self):
        first = self.execute()
        calls = self.adapter.calls
        reads = self.offline_ds.index_count
        second = self.execute()
        self.assertEqual(first, second)
        self.assertEqual(self.adapter.calls, calls)
        self.assertEqual(self.offline_ds.index_count, reads)

        partial_runner = E8ConfirmationRunner(
            config_path=self.config_path,
            ledger_path=self.ledger_path,
            output_dir=self.tmp / "partial",
            device="cpu",
        )
        partial = partial_runner.output_root / "synthetic_stream"
        partial.mkdir(parents=True)
        (partial / "initial_manifest.json").write_text("{}", encoding="utf-8")
        partial_runner.instantiate_model = mock.Mock(side_effect=AssertionError("must not instantiate"))
        with self.assertRaisesRegex(RuntimeError, "Refusing unsafe partial resume"):
            partial_runner.execute_stream(
                "synthetic_stream", "cotta_reset", 0, "order_a", "reset_per_event", event_subset=[520]
            )
        partial_runner.instantiate_model.assert_not_called()

    def test_identity_mismatch_fails_after_persistence_without_metric_record(self):
        self.offline_ds.records[1] = copy.deepcopy(self.offline_ds.records[1])
        self.offline_ds.records[1]["sample_id"] = "wrong_id"
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.execute()
        stream_dir = self.output_root / "synthetic_stream"
        self.assertTrue((stream_dir / "failure.json").is_file())
        self.assertFalse((stream_dir / "stream_summary.json").exists())

    def test_v2_ledger_counts_and_sealed_event_rejection(self):
        runner = E8ConfirmationRunner(
            REPO_ROOT / "configs/changeguard/e8_confirmation_matrix_v2.json",
            REPO_ROOT / "configs/changeguard/e8_execution_ledger_v2.json",
            device="cpu",
        )
        reused, execute_new = runner.planned_cells()
        self.assertEqual(len(reused), 2)
        self.assertEqual(len(execute_new), 67)
        self.assertEqual(len(reused) * 6 + len(execute_new) * 6, 414)
        certified = {c["stream_id"]: c["certification_status"] for c in reused}
        uncertified_historical = {
            c["stream_id"]: c["v1_certification_status"]
            for c in execute_new
            if c["method"] not in {"cotta_continual", "cotta_reset"}
        }
        uncertified_cotta = {
            c["stream_id"]: c["v1_certification_status"]
            for c in execute_new
            if c["method"] in {"cotta_continual", "cotta_reset"}
        }
        self.assertEqual(len(certified), 2)
        self.assertEqual(len(uncertified_historical), 55)
        self.assertEqual(len(uncertified_cotta), 12)
        self.assertEqual(set(certified.values()), {"CERTIFIED_POST_PERSISTENCE_GT_READ"})
        self.assertEqual(
            set(uncertified_historical.values()),
            {"GT_PRELOADED_BUT_GUARDED — NOT CERTIFIED"},
        )
        self.assertEqual(
            set(uncertified_cotta.values()),
            {"COMPUTATION COMPLETED, NOT CERTIFIED AS FORMAL EVIDENCE — EARLY GT DISK READ"},
        )
        runner.target_events = [520, 205]
        with self.assertRaisesRegex(RuntimeError, "Sealed test event 205"):
            runner.run_preflights()


if __name__ == "__main__":
    unittest.main()
