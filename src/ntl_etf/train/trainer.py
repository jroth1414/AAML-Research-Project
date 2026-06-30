"""Unified, deterministic training loop for every torch model (Phase C / Task M9).

Seeding + deterministic flags, AdamW, MSE on STANDARDIZED targets, gradient clipping, early
stopping on the validation fold with best-weight restore, CPU-first (AMP only if CUDA). The
PanelDataset already carries train-only ``(y_mu, y_sigma)`` per sample; ``predict`` de-standardizes
to original target units. num_workers=0 for Windows-safe determinism.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np

from ..data.panel import make_dataloader
from ..utils.seed import set_seed


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 32
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 15
    clip: float = 1.0
    seed: int = 1414
    max_steps: int | None = None  # for smoke tests


class Trainer:
    def __init__(self, train_cfg: TrainConfig):
        self.cfg = train_cfg

    def fit(self, module, train_ds, val_ds=None) -> dict:
        import torch

        set_seed(self.cfg.seed)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = module.to(device)
        opt = torch.optim.AdamW(
            module.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        train_loader = make_dataloader(
            train_ds, batch_size=self.cfg.batch_size, shuffle=True, seed=self.cfg.seed
        )
        best_val = float("inf")
        best_state = copy.deepcopy(module.state_dict())
        history = {"train": [], "val": []}
        bad = 0
        step = 0
        for _epoch in range(self.cfg.epochs):
            module.train()
            tr_losses = []
            for batch in train_loader:
                x = batch["x"].to(device)
                y = batch["y"].to(device)
                opt.zero_grad()
                vm = batch.get("var_mask")
                pred = module(x, vm.to(device) if vm is not None else None)
                loss = loss_fn(pred, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(module.parameters(), self.cfg.clip)
                opt.step()
                tr_losses.append(float(loss.item()))
                step += 1
                if self.cfg.max_steps and step >= self.cfg.max_steps:
                    break
            history["train"].append(float(np.mean(tr_losses)) if tr_losses else float("nan"))
            val_mse = self._eval(module, val_ds, device, loss_fn) if val_ds is not None else None
            history["val"].append(val_mse)
            if val_mse is not None and val_mse < best_val - 1e-9:
                best_val = val_mse
                best_state = copy.deepcopy(module.state_dict())
                bad = 0
            elif val_ds is not None:
                bad += 1
                if bad >= self.cfg.patience:
                    break
            if self.cfg.max_steps and step >= self.cfg.max_steps:
                break
        module.load_state_dict(best_state)
        history["best_val"] = best_val if val_ds is not None else None
        return history

    def _eval(self, module, ds, device, loss_fn) -> float:
        import torch

        if ds is None or len(ds) == 0:
            return float("nan")
        module.eval()
        losses = []
        loader = make_dataloader(ds, batch_size=self.cfg.batch_size, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                pred = module(batch["x"].to(device))
                losses.append(float(loss_fn(pred, batch["y"].to(device)).item()))
        return float(np.mean(losses)) if losses else float("nan")

    def predict(self, module, dataset) -> np.ndarray:
        """De-standardized predictions (raw target units), aligned to dataset order."""
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        module = module.to(device)
        module.eval()
        loader = make_dataloader(dataset, batch_size=self.cfg.batch_size, shuffle=False)
        out = []
        with torch.no_grad():
            for batch in loader:
                pred = module(batch["x"].to(device)).cpu().numpy()  # (B, H) standardized
                mu = batch["y_mu"].numpy().reshape(-1, 1)
                sigma = batch["y_sigma"].numpy().reshape(-1, 1)
                out.append(pred * sigma + mu)  # de-standardize
        return np.concatenate(out, axis=0) if out else np.empty((0, dataset.spec.horizon))
