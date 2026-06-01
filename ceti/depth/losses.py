"""Depth training losses for CETI whale domain adaptation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_scale_and_shift(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    """Least-squares scale and shift aligning prediction to target (per batch item)."""
    if prediction.ndim == 4:
        prediction = prediction.squeeze(1)
    if target.ndim == 4:
        target = target.squeeze(1)
    if mask.ndim == 4:
        mask = mask.squeeze(1)

    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    det = a_00 * a_11 - a_01 * a_01
    valid = det > 0

    scale = torch.zeros_like(b_0)
    shift = torch.zeros_like(b_1)
    scale[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    shift[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
    return scale, shift


class ScaleShiftInvariantLoss(nn.Module):
    """Align student depth to teacher pseudo-labels (scale-ambiguous relative depth)."""

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(prediction if prediction.ndim == 3 else prediction.squeeze(1))

        pred = prediction.squeeze(1) if prediction.ndim == 4 else prediction
        tgt = target.squeeze(1) if target.ndim == 4 else target
        m = (mask.squeeze(1) if mask.ndim == 4 else mask) > 0.5

        scale, shift = compute_scale_and_shift(pred, tgt, m.float())
        aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
        return F.l1_loss(aligned[m], tgt[m])


class GradientMatchingLoss(nn.Module):
    """Match spatial gradients between student and scale-aligned teacher depth."""

    def forward(self, prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        pred = prediction.squeeze(1) if prediction.ndim == 4 else prediction
        tgt = target.squeeze(1) if target.ndim == 4 else target
        if mask is None:
            mask = torch.ones_like(pred)

        m = (mask.squeeze(1) if mask.ndim == 4 else mask) > 0.5
        scale, shift = compute_scale_and_shift(pred, tgt, m.float())
        aligned = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)

        def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            dx = x[..., :, 1:] - x[..., :, :-1]
            dy = x[..., 1:, :] - x[..., :-1, :]
            return dx, dy

        pred_dx, pred_dy = grad(aligned)
        tgt_dx, tgt_dy = grad(tgt)
        m_dx = m[..., :, 1:] & m[..., :, :-1]
        m_dy = m[..., 1:, :] & m[..., :-1, :]
        loss_dx = F.l1_loss(pred_dx[m_dx], tgt_dx[m_dx])
        loss_dy = F.l1_loss(pred_dy[m_dy], tgt_dy[m_dy])
        return loss_dx + loss_dy
