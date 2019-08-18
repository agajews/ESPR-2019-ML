import torch
import torch.nn.functional as F
from torchvision import datasets, transforms

batch_size = 128
width = 100
learning_rate = 0.01
epochs = 100

class Network(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(28*28, width)
        self.layer2 = torch.nn.Linear(width, 10)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == '__main__':
    transform = transforms.Compose([
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True, transform=transform),
        batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transform),
        batch_size=batch_size, shuffle=True)

    net = Network()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    def get_accuracy():
        num_correct = 0
        num_total = 0
        for images, labels in test_loader:
            predictions = net(images)
            num_correct += (predictions.argmax(dim=1) == labels).sum().item()
            num_total += len(predictions)
        return num_correct / num_total

    for epoch in range(epochs):
        accuracy = get_accuracy()
        print('accuracy: {}'.format(accuracy))
        for batch_num, (images, labels) in enumerate(train_loader):
            predictions = net(images)
            loss = F.nll_loss(predictions, labels)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch_num % 100 == 0:
                print('epoch: {}, batch_num: {}, loss: {}'.format(
                    epoch, batch_num, loss.item()))
