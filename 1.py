import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = torch.Size(normalized_shape)
        self.eps = eps
        
        self.weight = nn.Parameter(torch.ones(self.normalized_shape))

    def forward(self, x):
        rms = torch.sqrt(
            torch.mean(x.pow(2), 
            dim=[-(i+1) for i in range(len(self.normalized_shape))],
            keepdim=True
        ) + self.eps)
        
        return self.weight * (x / rms)
    
custom_norm1 = RMSNorm(128)
norm1 = nn.RMSNorm(128)
x1 = torch.randn(32, 128)
out1 = norm1(x1)
custom_out1 = custom_norm1(x1)

custom_norm2 = RMSNorm((64, 64))
norm2 = nn.RMSNorm((64, 64))

x2 = torch.randn(16, 3, 64, 64)
custom_out2 = custom_norm2(x2)
out2 = norm2(x2)

assert torch.allclose(out1, custom_out1, atol=1e-6), '1D outputs mismatch'
assert torch.allclose(out2, custom_out2, atol=1e-6), '2D outputs mismatch'