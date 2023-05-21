import torch
import torch.nn.functional as F


def masked_l1(disp_pred: torch.Tensor, disp_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(disp_pred[mask], disp_target[mask], reduction="mean")


def threshold_error(
    disp_pred: torch.Tensor, disp_target: torch.Tensor, mask: torch.Tensor, threshold: float
) -> torch.Tensor:
    disp_pred, disp_target = disp_pred[mask], disp_target[mask]
    error = torch.abs(disp_pred - disp_target)
    return torch.mean((error > threshold).float())


def end_point_error(disp_pred: torch.Tensor, disp_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    disp_pred, disp_target = disp_pred[mask], disp_target[mask]
    return F.l1_loss(disp_pred, disp_target, reduction="mean")


def d1_error(disp_pred: torch.Tensor, disp_target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    disp_pred, disp_target = disp_pred[mask], disp_target[mask]
    error = torch.abs(disp_pred - disp_target)
    return torch.mean(((error > 3) & (error / disp_target.abs() > 0.05)).float())
