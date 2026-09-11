"""Validation and normalization of the Search-R1 generated-token mask."""

from __future__ import annotations

from typing import Mapping, Optional

import torch


def validate_generated_token_mask(
    response_mask: torch.Tensor,
    *,
    expected_shape: Optional[tuple[int, int]] = None,
    response_attention_mask: Optional[torch.Tensor] = None,
    name: str = "response_mask",
) -> torch.Tensor:
    """Return a detached boolean mask for model-generated response tokens.

    Search-R1's ``ToolAgentLoop`` uses one response-length tensor for the full
    trajectory.  It appends ``1`` for every token emitted by the model and
    ``0`` for tool observations and padding.  Observations therefore remain
    in the model prefix while never becoming OPD actions.

    ``response_attention_mask`` is optional defensive padding protection.  It
    may cover exactly the response or the full sequence; in the latter case
    its trailing response-length slice is used.  No token-id heuristic is
    applied, because an observation can contain arbitrary vocabulary tokens.
    """

    if not isinstance(response_mask, torch.Tensor) or response_mask.ndim != 2:
        shape = getattr(response_mask, "shape", None)
        raise ValueError(
            f"{name} must have shape [batch, response_length], got {shape}"
        )
    if expected_shape is not None and tuple(response_mask.shape) != tuple(expected_shape):
        raise ValueError(
            f"{name} shape does not match expected shape: "
            f"mask={tuple(response_mask.shape)} expected={tuple(expected_shape)}"
        )

    detached = response_mask.detach()
    if detached.is_floating_point() and not torch.isfinite(detached).all():
        raise ValueError(f"{name} contains non-finite values")
    is_binary = (detached == 0) | (detached == 1)
    if not bool(is_binary.all()):
        raise ValueError(f"{name} must contain only 0/1 values")
    mask = detached.to(dtype=torch.bool)

    if response_attention_mask is not None:
        if (
            not isinstance(response_attention_mask, torch.Tensor)
            or response_attention_mask.ndim != 2
            or response_attention_mask.shape[0] != mask.shape[0]
        ):
            raise ValueError(
                "response_attention_mask must have the same batch dimension "
                f"as {name}: attention={getattr(response_attention_mask, 'shape', None)} "
                f"mask={tuple(mask.shape)}"
            )
        if response_attention_mask.shape[1] == mask.shape[1]:
            attention = response_attention_mask
        elif response_attention_mask.shape[1] > mask.shape[1]:
            attention = response_attention_mask[:, -mask.shape[1] :]
        else:
            raise ValueError(
                "response_attention_mask is shorter than the response: "
                f"attention={tuple(response_attention_mask.shape)} "
                f"response={tuple(mask.shape)}"
            )
        if attention.device != mask.device:
            attention = attention.to(mask.device)
        mask = mask & attention.detach().to(dtype=torch.bool)
    return mask


def select_opd_response_mask(
    batch: Mapping[str, torch.Tensor],
    *,
    expected_shape: tuple[int, int],
    response_attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select an explicit OPD mask, falling back to Search-R1's mask.

    ``opd_response_mask`` is reserved for future rollout adapters that
    propagate an independently named mask.  The current Search-R1 loop
    already supplies the same contract as ``response_mask``.
    """

    if "opd_response_mask" in batch:
        key = "opd_response_mask"
    elif "response_mask" in batch:
        key = "response_mask"
    else:
        raise KeyError(
            "OPD batch must contain response_mask or opd_response_mask"
        )
    return validate_generated_token_mask(
        batch[key],
        expected_shape=expected_shape,
        response_attention_mask=response_attention_mask,
        name=key,
    )

