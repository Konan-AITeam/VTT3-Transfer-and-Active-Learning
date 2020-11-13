import torch


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        grad_input = - 0.1 * grad_output.clone()
        return grad_input


class GradZero(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        # print('backward')
        grad_input = - 0 * grad_output.clone()
        return grad_input