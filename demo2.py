import torch


def f(x):
    return (x - 5) ** 2


x = torch.tensor(6.0, requires_grad=True, dtype=torch.float64)

alpha = torch.tensor(0.01, dtype=torch.float16)

for _ in range(1000):
    obj = f(x)
    obj.backward()
    with torch.no_grad():
        x -= alpha * x.grad
        x.grad.zero_()
    print(x.item())
