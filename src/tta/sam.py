"""
Sharpness-Aware Minimization (SAM) optimizer for ChangeGuard test-time adaptation.

Upstream Provenance:
- Repository: https://github.com/mr-eggplant/SAR (ICLR 2023 Oral)
- Commit: 20f6e24b17525f34503510afccedc0629b67b7c4
- License: BSD 3-Clause License (SAR Authors)
- Upstream Base: https://github.com/davda54/sam (MIT License)

Safety Enhancement:
- Implements `restore_perturbation(zero_grad=True)` to safely revert parameters
  from theta + epsilon(theta) back to theta if the second SAM stage cannot proceed
  (e.g., empty stage-2 sample selection or numerical anomaly).
"""

from __future__ import annotations

import torch
from typing import Any, Callable, Dict, Optional


class SAM(torch.optim.Optimizer):
    """Sharpness-Aware Minimization optimizer.

    Climbs to a local loss maximum within radius rho in the first step,
    evaluates gradients at that perturbed state, and steps the base optimizer
    from the original weights in the direction of flatter loss basins.
    """

    def __init__(
        self,
        params,
        base_optimizer: Callable[..., torch.optim.Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ):
        if rho < 0.0:
            raise ValueError(f"Invalid rho, should be non-negative: {rho}")

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)
        self.perturbed = False

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> None:
        """Perturbs parameters to the local maximum within the rho ball."""
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)

            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (
                    (torch.pow(p, 2) if group["adaptive"] else 1.0)
                    * p.grad
                    * scale.to(p)
                )
                p.add_(e_w)  # climb to the local maximum "w + e(w)"

        self.perturbed = True
        if zero_grad:
            self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> bool:
        """Restores parameters from w + e(w) back to w and applies base optimizer update.

        Guarantees that EVERY parameter perturbed in first_step is restored back to its
        pre-perturbation value, even if p.grad is None in the second pass or an anomaly occurs.

        Returns:
            applied_step: True if base optimizer stepped on finite gradients; False otherwise.
        """
        applied_step = False
        try:
            # 1. Restore ALL perturbed parameters back to their original weights
            for group in self.param_groups:
                for p in group["params"]:
                    if "old_p" in self.state[p]:
                        p.data.copy_(self.state[p]["old_p"])

            # 2. Check if gradients are finite before stepping base optimizer
            all_finite = True
            has_grad = False
            for group in self.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        has_grad = True
                        if not torch.isfinite(p.grad).all():
                            all_finite = False
                            break
                if not all_finite:
                    break

            if all_finite and has_grad:
                self.base_optimizer.step()
                applied_step = True
        finally:
            self.perturbed = False
            for group in self.param_groups:
                for p in group["params"]:
                    if "old_p" in self.state[p]:
                        del self.state[p]["old_p"]
            if zero_grad:
                self.zero_grad()
        return applied_step

    @torch.no_grad()
    def restore_perturbation(self, zero_grad: bool = False) -> None:
        """Safety restoration: restores w from w + e(w) without applying an optimizer step.

        Used when the second SAM stage cannot proceed (e.g. empty stage-2 sample selection,
        non-finite loss, or execution exception).
        """
        if not self.perturbed:
            if zero_grad:
                self.zero_grad()
            return

        try:
            for group in self.param_groups:
                for p in group["params"]:
                    if "old_p" in self.state[p]:
                        p.data.copy_(self.state[p]["old_p"])
        finally:
            self.perturbed = False
            for group in self.param_groups:
                for p in group["params"]:
                    if "old_p" in self.state[p]:
                        del self.state[p]["old_p"]
            if zero_grad:
                self.zero_grad()

    def _grad_norm(self) -> torch.Tensor:
        shared_device = self.param_groups[0]["params"][0].device
        grad_list = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is not None:
                    g = (torch.abs(p) if group["adaptive"] else 1.0) * p.grad
                    grad_list.append(g.norm(p=2).to(shared_device))

        if not grad_list:
            return torch.tensor(0.0, device=shared_device)

        return torch.norm(torch.stack(grad_list), p=2)

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        self.perturbed = False

