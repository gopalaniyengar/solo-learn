import torchvision.models as models
from torch.nn import Parameter
import torch
import torch.nn as nn
from torch.autograd import Function

import torch.nn.functional as F
from torch.autograd import Variable


class GradientReversalFunction(Function):

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grads):
        lambda_ = ctx.lambda_
        lambda_ = grads.new_tensor(lambda_)
        dx = -lambda_ * grads
        return dx, None

class GradientReversal(torch.nn.Module):
    def __init__(self, lambda_ = 1):
        super(GradientReversal, self).__init__()
        self.lambda_ = lambda_

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


def reverse_grad(x, lambd=1.0):
    return GradientReversal(lambd)(x)
