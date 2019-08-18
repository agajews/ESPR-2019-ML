import torch

matrix = torch.randn(2, 3)


def matmul(x):
    return matrix.matmul(x)


class MatMul(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = torch.randn(2, 3)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        return self.matrix.matmul(x)


mod1 = MatMul()
mod2 = MatMul()

x = torch.randn(3, 2)
y = mod(x)
print(y)
