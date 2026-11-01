"""Deterministic event-balanced epoch sampler (Component M1)."""

import hashlib
import random
from typing import Any, Dict, Iterator, List, Optional
import numpy as np
from torch.utils.data import Sampler


def compute_m1_seed(run_seed: int, epoch: int) -> int:
    """Computes deterministic M1 epoch sampler seed."""
    return (int(run_seed) * 1000003 + int(epoch) * 997 + 101) & 0xFFFFFFFF


class EventBalancedSampler(Sampler[int]):
    """Event-balanced epoch sampler ensuring uniform event representation.

    Semantics:
    1. Epoch length equals N (total eligible patches across all active training events).
    2. Draws per event are approximately uniform (floor(N/E) or ceil(N/E)).
    3. Within each event, samples are drawn without replacement from a shuffled queue,
       recycling only when the event's patches are exhausted.
    4. Events are interleaved round-by-round to avoid event clustering.
    5. Fully deterministic given run_seed and epoch index.
    """

    def __init__(
        self,
        samples_metadata: List[Dict[str, Any]],
        run_seed: int,
        epoch: int = 0,
        start_index: int = 0,
    ):
        """Initializes EventBalancedSampler.

        Args:
            samples_metadata: List of dicts, each containing at least 'event_id' and 'sample_id'.
                The index in this list corresponds to dataset item index.
            run_seed: Base run seed.
            epoch: Current epoch index (0-based).
            start_index: Offset for resuming within an epoch.
        """
        self.samples = samples_metadata
        self.total_samples = len(samples_metadata)
        self.run_seed = int(run_seed)
        self.epoch = int(epoch)
        self.start_index = int(start_index)
        self.sampler_seed = compute_m1_seed(self.run_seed, self.epoch)

        # Group indices by event_id
        self.event_to_indices: Dict[int, List[int]] = {}
        for idx, item in enumerate(self.samples):
            eid = int(item["event_id"])
            if eid not in self.event_to_indices:
                self.event_to_indices[eid] = []
            self.event_to_indices[eid].append(idx)

        self.event_ids = sorted(list(self.event_to_indices.keys()))
        self.num_events = len(self.event_ids)

        if self.num_events == 0:
            raise ValueError("EventBalancedSampler received empty training events list.")

        # Compute exact quota per event
        # N total draws across E events:
        # Base quota Q = N // E, remainder R = N % E
        # First R events (sorted by event_id) receive Q + 1, remaining receive Q
        base_quota = self.total_samples // self.num_events
        remainder = self.total_samples % self.num_events

        self.expected_draws_per_event: Dict[int, int] = {}
        for rank, eid in enumerate(self.event_ids):
            quota = base_quota + (1 if rank < remainder else 0)
            self.expected_draws_per_event[eid] = quota

        # Generate the full epoch sequence of dataset indices
        self.epoch_indices = self._generate_epoch_indices()

    def _generate_epoch_indices(self) -> List[int]:
        """Deterministically generates the N sample indices for this epoch."""
        rng = random.Random(self.sampler_seed)

        # 1. Initialize per-event shuffled queues
        per_event_pools: Dict[int, List[int]] = {}
        for eid in self.event_ids:
            pool = list(self.event_to_indices[eid])
            rng.shuffle(pool)
            per_event_pools[eid] = pool

        # 2. Interleave events round-by-round to distribute events uniformly
        max_quota = max(self.expected_draws_per_event.values())
        epoch_indices: List[int] = []

        # Track remaining draws per event
        remaining_draws = dict(self.expected_draws_per_event)

        for _ in range(max_quota):
            # Candidate events for this round
            active_round_events = [eid for eid in self.event_ids if remaining_draws[eid] > 0]
            # Shuffle event draw order within round for diversity
            rng.shuffle(active_round_events)

            for eid in active_round_events:
                pool = per_event_pools[eid]
                if len(pool) == 0:
                    # Recycle event samples: reshuffle
                    pool = list(self.event_to_indices[eid])
                    rng.shuffle(pool)
                    per_event_pools[eid] = pool

                sample_idx = pool.pop(0)
                epoch_indices.append(sample_idx)
                remaining_draws[eid] -= 1

        if len(epoch_indices) != self.total_samples:
            raise RuntimeError(
                f"Generated epoch length {len(epoch_indices)} != expected {self.total_samples}"
            )

        return epoch_indices

    def __iter__(self) -> Iterator[int]:
        for idx in self.epoch_indices[self.start_index:]:
            yield idx

    def __len__(self) -> int:
        return max(0, self.total_samples - self.start_index)

    def get_epoch_summary(self) -> Dict[str, Any]:
        """Computes audit metrics and sample digest for the epoch."""
        realized_draws: Dict[int, int] = {eid: 0 for eid in self.event_ids}
        unique_samples: Dict[int, set] = {eid: set() for eid in self.event_ids}
        yielded_sample_ids: List[str] = []

        for idx in self.epoch_indices:
            item = self.samples[idx]
            eid = int(item["event_id"])
            sid = str(item["sample_id"])
            realized_draws[eid] += 1
            unique_samples[eid].add(sid)
            yielded_sample_ids.append(sid)

        repeat_counts = {
            eid: realized_draws[eid] - len(unique_samples[eid])
            for eid in self.event_ids
        }

        # SHA-256 digest of exact yielded sample sequence
        digest = hashlib.sha256(",".join(yielded_sample_ids).encode("utf-8")).hexdigest()

        draw_counts = list(realized_draws.values())
        return {
            "epoch_index": self.epoch,
            "sampler_seed": self.sampler_seed,
            "total_draws": len(self.epoch_indices),
            "expected_draws_per_event": self.expected_draws_per_event,
            "realized_draws_per_event": realized_draws,
            "unique_samples_used_per_event": {eid: len(unique_samples[eid]) for eid in self.event_ids},
            "repeat_count_per_event": repeat_counts,
            "min_event_draws": min(draw_counts),
            "max_event_draws": max(draw_counts),
            "ordered_sample_id_digest": digest,
        }
