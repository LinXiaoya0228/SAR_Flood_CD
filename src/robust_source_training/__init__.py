"""Robust Source Training (RST) package for ChangeGuard."""

from src.robust_source_training.contracts import (
    AUTHORITATIVE_SOURCE_EVENTS,
    FORBIDDEN_TARGET_EVENTS,
    ContractViolationError,
    validate_event_contracts,
)
from src.robust_source_training.dataset import (
    RSTR0SourceDataset,
    build_r0_dataloader,
    seed_everything,
)
from src.robust_source_training.sampler import EventBalancedSampler
from src.robust_source_training.loss import (
    RobustSourceLoss,
    compute_training_class_weights,
)
from src.robust_source_training.augmentation import SynchronizedPairedSARAugmentation
from src.robust_source_training.evaluation import EventIndependentEvaluator
from src.robust_source_training.checkpoint import (
    assert_states_equal,
    load_rst_checkpoint,
    save_rst_checkpoint,
)
from src.robust_source_training.manifest import (
    write_final_manifest,
    write_preflight_manifest,
    write_terminal_status,
)
from src.robust_source_training.runner import RSTExecutionEngine

__version__ = "1.0.0"


