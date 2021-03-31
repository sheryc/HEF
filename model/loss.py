import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target.type_as(output))


def ce_loss(output, target):
    return F.cross_entropy(output, target)


def bce_loss(output, target):
    """
    output: a (batch_size, 1) tensor
    target: a (batch_size, ) tensor of dtype int
    
    """
    return F.binary_cross_entropy_with_logits(output.squeeze(), target.type_as(output))


def abl_bce_loss_for_stopper(output, target):
    return F.binary_cross_entropy_with_logits(output[:, 1].squeeze(), (target == 1).type_as(output))


def info_nce_loss(output, target):
    """
    output: a (batch_size, 1+negative_size) tensor
    target: a (batch_size, ) tensor of dtype long, all zeros
    """
    return F.cross_entropy(output, target, reduction="sum")


def mt_loss(pred_pathfinder, pred_stopper, tgt_pathfinder, tgt_stopper, pathfinder_loss_fn, stopper_loss_fn, eta):
    return eta * pathfinder_loss_fn(pred_pathfinder, tgt_pathfinder) + (1 - eta) * stopper_loss_fn(
        pred_stopper, tgt_stopper)
