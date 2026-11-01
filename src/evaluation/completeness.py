"""Evaluation completeness verification module.

Ensures that every evaluation run processes exactly the expected samples from the manifest
without omissions, duplicates, or silent truncations.
"""

from typing import Dict, Any, List, Optional


class CompletenessTracker:
    """Tracks and asserts the completeness of an evaluation run across events."""

    def __init__(
        self,
        expected_samples_per_event: Dict[int, List[str]],
        allow_partial: bool = False
    ):
        """
        Args:
            expected_samples_per_event: Mapping from event_id to list of expected sample_ids.
            allow_partial: If True, do not raise exception on incomplete/duplicated runs (diagnostic mode).
        """
        self.expected_samples: Dict[int, List[str]] = {
            int(eid): list(sids) for eid, sids in expected_samples_per_event.items()
        }
        self.allow_partial = allow_partial
        self.processed_samples: Dict[int, List[str]] = {
            int(eid): [] for eid in self.expected_samples
        }
        self.valid_pixels: Dict[int, int] = {
            int(eid): 0 for eid in self.expected_samples
        }
        self.batch_count = 0

    def update(
        self,
        event_ids: List[int],
        sample_ids: List[str],
        valid_pixel_counts: Optional[List[int]] = None
    ):
        """Update tracker with a processed batch."""
        self.batch_count += 1
        for i, (eid, sid) in enumerate(zip(event_ids, sample_ids)):
            eid = int(eid)
            if eid not in self.processed_samples:
                raise ValueError(f"Encountered unexpected event_id {eid} not in expected set!")
            self.processed_samples[eid].append(str(sid))
            if valid_pixel_counts is not None:
                self.valid_pixels[eid] += int(valid_pixel_counts[i])

    def compute_report(self) -> Dict[str, Any]:
        """Compute full completeness diagnostics across all events."""
        event_reports = {}
        total_expected = sum(len(s) for s in self.expected_samples.values())
        total_processed = sum(len(s) for s in self.processed_samples.values())
        all_passed = True
        error_messages = []

        for eid, exp_ids in sorted(self.expected_samples.items()):
            proc_ids = self.processed_samples[eid]
            exp_set = set(exp_ids)
            proc_set = set(proc_ids)

            # Find duplicates
            seen = set()
            duplicates = []
            for s in proc_ids:
                if s in seen:
                    duplicates.append(s)
                seen.add(s)

            missing = sorted(list(exp_set - proc_set))
            unexpected = sorted(list(proc_set - exp_set))
            count_match = (len(proc_ids) == len(exp_ids))

            status = "PASS"
            reasons = []
            if not count_match:
                status = "FAIL"
                reasons.append(f"count mismatch: expected {len(exp_ids)}, processed {len(proc_ids)}")
            if duplicates:
                status = "FAIL"
                reasons.append(f"{len(duplicates)} duplicate sample IDs detected")
            if missing:
                status = "FAIL"
                reasons.append(f"{len(missing)} missing sample IDs")
            if unexpected:
                status = "FAIL"
                reasons.append(f"{len(unexpected)} unexpected sample IDs")

            if status == "FAIL":
                all_passed = False
                error_messages.append(f"Event {eid} FAILED: " + "; ".join(reasons))

            event_reports[str(eid)] = {
                "event_id": eid,
                "expected_sample_count": len(exp_ids),
                "processed_sample_count": len(proc_ids),
                "duplicate_sample_count": len(duplicates),
                "missing_sample_count": len(missing),
                "unexpected_sample_count": len(unexpected),
                "total_valid_pixels": self.valid_pixels[eid],
                "status": status,
                "failure_reasons": reasons,
            }

        return {
            "status": "PASS" if all_passed else "FAIL",
            "total_expected_samples": total_expected,
            "total_processed_samples": total_processed,
            "processed_batch_count": self.batch_count,
            "error_messages": error_messages,
            "events": event_reports,
        }

    def validate_or_raise(self) -> Dict[str, Any]:
        """Validate completeness; fail loudly if incomplete unless allow_partial is True."""
        report = self.compute_report()
        if report["status"] == "FAIL" and not self.allow_partial:
            msg = "FATAL: Evaluation completeness check failed!\n" + "\n".join(report["error_messages"])
            raise RuntimeError(msg)
        return report
