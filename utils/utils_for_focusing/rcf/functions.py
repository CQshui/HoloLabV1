import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
# loss function

def cross_entropy_loss_RCF(prediction, label):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask==1).float()).float()
    num_negative = torch.sum((mask==0).float()).float()

    mask[mask == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[mask == 0] = 1.1 * num_positive / (num_positive + num_negative)
    mask[mask == 2] = 0
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(), weight=mask, reduce=False)
    return torch.sum(cost)


def dice_loss(prediction, label, smooth=1e-5):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    # 计算正负样本的权重
    weight_positive = 1.0 * num_negative / (num_positive + num_negative)
    weight_negative = 1.2 * num_positive / (num_positive + num_negative)

    # 动态权重掩码
    mask[mask == 1] = weight_positive
    mask[mask == 0] = weight_negative
    mask[mask == 2] = 0  # 忽略标签为2的区域

    # 计算交集和并集
    intersection = torch.sum(prediction * label.float())
    dice = (2. * intersection + smooth) / (torch.sum(prediction) + torch.sum(label.float()) + smooth)

    return 1 - dice  # 返回1减去Dice系数，作为损失值


def focal_loss(prediction, label, gamma=2, alpha=0.25, smooth=1e-5):
    label = label.long()
    mask = label.float()
    num_positive = torch.sum((mask == 1).float()).float()
    num_negative = torch.sum((mask == 0).float()).float()

    # 计算正负样本的权重
    weight_positive = alpha * (num_negative / (num_positive + num_negative)) ** gamma
    weight_negative = (1 - alpha) * (num_positive / (num_positive + num_negative)) ** gamma

    # 动态权重掩码
    mask[mask == 1] = weight_positive
    mask[mask == 0] = weight_negative
    mask[mask == 2] = 0  # 忽略标签为2的区域

    # 计算Focal Loss
    pt = torch.where(label == 1, prediction, 1 - prediction)
    focal_weight = torch.pow(1 - pt, gamma)
    loss = -focal_weight * torch.log(torch.clamp(pt, smooth, 1 - smooth))

    return torch.sum(loss)

class SGD_caffe(Optimizer):
    r"""Implements stochastic gradient descent (optionally with momentum).

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
        momentum (float, optional): momentum factor (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        dampening (float, optional): dampening for momentum (default: 0)
        nesterov (bool, optional): enables Nesterov momentum (default: False)

    Example:
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et. al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
                  v = \rho * v + g \\
                  p = p - lr * v

        where p, g, v and :math:`\rho` denote the parameters, gradient,
        velocity, and momentum respectively.

        This is in contrast to Sutskever et. al. and
        other frameworks which employ an update of the form

        .. math::
             v = \rho * v + lr * g \\
             p = p - v

        The Nesterov version is analogously modified.
    """

    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_caffe, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_caffe, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(group['lr'], d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(group['lr'], d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-1, d_p)

        return loss
