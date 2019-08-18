import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

batch_size = 100
width = 100
learning_rate = 0.01
num_epochs = 10


class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28 * 28, width)
        self.layer2 = torch.nn.Linear(width, 10)

    def forward(self, x):
        x = x.reshape(-1, 28 * 28)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.log_softmax(x)
        return x


transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=True, download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST("../data", train=False, transform=transform),
    batch_size=batch_size,
    shuffle=True,
)

net = Network()
optim = torch.optim.SGD(net.parameters(), lr=learning_rate)


def get_accuracy(loader):
    num_correct = 0
    num_total = 0
    for images, labels in loader:
        predictions = net(images)
        bools = predictions.argmax(dim=1) == labels
        num_correct += bools.sum().item()
        num_total += len(predictions)
    return num_correct / num_total


train_accuracies = [get_accuracy(train_loader)]
test_accuracies = [get_accuracy(test_loader)]

for epoch in range(num_epochs):
    train_accuracy = get_accuracy(train_loader)
    test_accuracy = get_accuracy(test_loader)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)

    print(
        "epoch: {}, train accuracy: {}, test accuracy: {}".format(
            epoch, train_accuracy, test_accuracy
        )
    )

    for t, (images, labels) in enumerate(train_loader):
        predictions = net(images)

        loss = F.nll_loss(predictions, labels)
        loss.backward()
        optim.step()
        optim.zero_grad()
        if t % 100 == 0:
            print(loss.item())

plt.plot(train_accuracies, label="train")
plt.plot(test_accuracies, label="test")
plt.legend()
plt.show()
