from __future__ import annotations
from typing import Dict, Tuple
import torch
from torch.nn import functional as F

TWEEDIE_SOFTPLUS_BETA = 50.0
TWEEDIE_EPS = 1e-6


def tweedie_cpe_params(
    m_old: torch.Tensor, z_old: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Map raw head outputs to (m_pos, lam, p_pos, theta)."""
    m_pos = F.softplus(m_old, beta=TWEEDIE_SOFTPLUS_BETA) + TWEEDIE_EPS
    lam = F.softplus(z_old) + TWEEDIE_EPS
    p_pos = -torch.expm1(-lam)
    mu = p_pos * m_pos
    theta = mu / lam
    return (m_pos, lam, p_pos, theta)


def tweedie_cpe_p15_loglik(
    y: torch.Tensor, lam: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    """log P(Y=y) for the p=1.5 compound Poisson-Exponential; y >= 0."""
    in_dtype = y.dtype
    lam = lam.to(torch.float64).clamp(1e-06, 20.0)
    theta = theta.to(torch.float64).clamp(0.0001, 1000.0)
    y = y.to(torch.float64).clamp_min(0.0)
    pos = y > 0
    yp = y.clamp_min(1e-12)
    x = (lam * yp / theta).clamp_min(1e-20)
    z = 2.0 * torch.sqrt(x)
    log_i1 = torch.where(
        z < 0.0001,
        torch.log(z.clamp_min(1e-20) * 0.5),
        torch.log(torch.special.i1e(z).clamp_min(1e-300)) + z.abs(),
    )
    positive_out = (
        -lam
        - yp / theta
        + 0.5 * (torch.log(lam) - torch.log(theta) - torch.log(yp))
        + log_i1
    )
    out = torch.where(pos, positive_out, -lam)
    return out.to(in_dtype)


def tweedie_cpe_p15_nll(
    y: torch.Tensor, lam: torch.Tensor, theta: torch.Tensor
) -> torch.Tensor:
    return -tweedie_cpe_p15_loglik(y, lam, theta)


def tweedie_cpe_weighted_nll(
    pred: torch.Tensor,
    presence_logit: torch.Tensor,
    target: torch.Tensor,
    target_mask: torch.Tensor,
    *,
    positive_threshold: float,
    pos_pool_count: torch.Tensor,
    zero_pool_count: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Inverse-sampling-corrected Tweedie NLL.

    The sampler draws K_pos positives and K_zero zeros per cell from the full
    masked-candidate pools of size N_pos / N_zero.  Entries are reweighted by
    w = N_pool / K_sampled and each cell is normalized by its full candidate
    count, so the expected loss matches the natural measured-expression
    distribution instead of the 1:1 sampled ratio.

    pred/presence_logit/target/target_mask: (B, K); pos_pool_count/
    zero_pool_count: (B,) per-cell candidate pool sizes from the sampler.
    Returns (scalar loss, diagnostics).
    """
    m_pos, lam, p_pos, theta = tweedie_cpe_params(pred, presence_logit)
    mask = target_mask.to(dtype=torch.bool)
    nll = tweedie_cpe_p15_nll(target.clamp_min(0.0), lam, theta)
    is_pos = mask & (target > positive_threshold)
    k_pos = is_pos.sum(dim=1).clamp(min=1).to(torch.float64)
    k_zero = (mask & ~is_pos).sum(dim=1).clamp(min=1).to(torch.float64)
    n_pos = pos_pool_count.to(torch.float64).clamp(min=1.0)
    n_zero = zero_pool_count.to(torch.float64).clamp(min=1.0)
    w = torch.where(
        is_pos, (n_pos / k_pos).unsqueeze(1), (n_zero / k_zero).unsqueeze(1)
    )
    w = w.to(nll.dtype)
    per_cell = (nll * w * mask).sum(dim=1) / (n_pos + n_zero)
    valid_cells = mask.any(dim=1) & (n_pos + n_zero > 0)
    valid_cells_f = valid_cells.to(dtype=per_cell.dtype)
    loss = (per_cell * valid_cells_f).sum() / valid_cells_f.sum().clamp(min=1.0)
    with torch.no_grad():
        lam32 = lam.to(torch.float32)
        mask_f = mask.to(dtype=torch.float32)
        mask_count = mask_f.sum().clamp(min=1.0)

        def masked_mean(value: torch.Tensor) -> torch.Tensor:
            value32 = value.to(dtype=torch.float32)
            return (value32 * mask_f).sum() / mask_count

        diag = {
            "tweedie_nll": loss.detach().to(torch.float32),
            "tweedie_lambda_mean": masked_mean(lam32),
            "tweedie_theta_mean": masked_mean(theta),
            "tweedie_p_pos_mean": masked_mean(p_pos),
            "tweedie_m_pos_mean": masked_mean(m_pos),
        }
    return (loss, diag)
