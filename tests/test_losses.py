import torch

from polypseg.losses import BCEDiceLoss, bce_dice_loss, soft_dice_loss


def _target():
    t = torch.zeros(2, 1, 16, 16)
    t[:, :, 4:12, 4:12] = 1.0
    return t


def test_dice_loss_is_zero_for_perfect_and_one_for_inverted():
    t = _target()
    confident = (t * 2 - 1) * 20  # +20 logits on polyp, -20 elsewhere
    assert soft_dice_loss(confident, t).item() < 0.01
    assert soft_dice_loss(-confident, t).item() > 0.99


def test_combined_loss_interpolates_and_is_differentiable():
    t = _target()
    logits = torch.zeros_like(t, requires_grad=True)
    loss = bce_dice_loss(logits, t, 0.5)
    bce_only, dice_only = bce_dice_loss(logits, t, 1.0), bce_dice_loss(logits, t, 0.0)
    assert abs(loss.item() - 0.5 * (bce_only.item() + dice_only.item())) < 1e-6
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.isclose(BCEDiceLoss(0.5)(logits.detach(), t), loss.detach())
