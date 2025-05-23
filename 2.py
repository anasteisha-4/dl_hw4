import torch
from torch.autograd import Function

class AutogradFunction(Function):
    @staticmethod
    def forward(ctx, X, Y):
        exp_X = torch.exp(X)
        cos_Y = torch.cos(Y)
        ctx.save_for_backward(exp_X, Y)
        
        result = exp_X + cos_Y
        return result

    @staticmethod
    def backward(ctx, grad_output):
        exp_X, Y = ctx.saved_tensors
        
        grad_X = grad_output * exp_X
        grad_Y = -grad_output * torch.sin(Y)
        
        return grad_X, grad_Y

X = torch.tensor([1.0, 2.0], requires_grad=True)
Y = torch.tensor([0.5, 1.0], requires_grad=True)

custom_output = AutogradFunction.apply(X, Y)
output = torch.exp(X) + torch.cos(Y)

assert torch.allclose(custom_output, output), "Forward pass mismatch"
print(custom_output)
print(output)

custom_grad_X, custom_grad_Y = torch.autograd.grad(
    outputs=custom_output.sum(),
    inputs=[X, Y],
    retain_graph=True
)

grad_X, grad_Y = torch.autograd.grad(
    outputs=output.sum(),
    inputs=[X, Y],
    retain_graph=True
)

assert torch.allclose(custom_grad_X, grad_X), "X gradient mismatch"
assert torch.allclose(custom_grad_Y, grad_Y), "Y gradient mismatch"
print(custom_grad_X, custom_grad_Y)
print(grad_X, grad_Y)