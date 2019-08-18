import torch
import matplotlib.pyplot as plt
import numpy as np


nsteps = 10000


def f(x, y):
    return (x - 5) ** 2 + (y - 6) ** 2


x = torch.tensor(10.0, requires_grad=True)
y = torch.tensor(10.0, requires_grad=True)

optim = torch.optim.Adam([x, y], lr=1)

xs = []
ys = []

for _ in range(nsteps):
    obj = f(x, y)
    obj.backward()
    optim.step()
    optim.zero_grad()
    print(x.item(), y.item())
    xs.append(x.item())
    ys.append(y.item())

plt.scatter(xs, ys, c=np.log(np.arange(nsteps)))
plt.show()
