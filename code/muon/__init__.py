"""Muon optimizer integration for Search-OPD."""

from .muon import Muon


class SearchOPDMuon(Muon):
    """Muon with automatic Muon/AdamW parameter partitioning for Verl."""

    def __init__(self, params, lr=1e-3, weight_decay=0.1, **kwargs):
        params = list(params)
        muon_params = [p for p in params if p.requires_grad and p.ndim == 2]
        adamw_params = [p for p in params if p.requires_grad and p.ndim != 2]
        super().__init__(
            lr=lr,
            wd=weight_decay,
            muon_params=muon_params,
            adamw_params=adamw_params,
            **kwargs,
        )


__all__ = ["Muon", "SearchOPDMuon"]
