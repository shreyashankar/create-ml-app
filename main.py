"""Example script to train a simple FC NN on MNIST."""

import argparse

import torch
import torchvision

# Parse arguments
parser = argparse.ArgumentParser(description='Parameters.')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--num_epochs', type=int, default=15)
parser.add_argument('--input_dim', type=int, default=784)
parser.add_argument('--hidden_dim', type=int, default=128)
parser.add_argument('--output_dim', type=int, default=10)
args = parser.parse_args()
RANDOM_SEED = args.seed
NUM_EPOCHS = args.num_epochs
INPUT_DIM = args.input_dim
HIDDEN_DIM = args.hidden_dim
OUTPUT_DIM = args.output_dim

# Set random seed
torch.manual_seed(RANDOM_SEED)

# Load dataset and transforms
transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                            torchvision.transforms.Normalize(
                                                (0.5,), (0.5,)),
                                            ])
train_set = torchvision.datasets.MNIST(
    '.data/train', download=True, train=True, transform=transform)
test_set = torchvision.datasets.MNIST(
    '.data/test', download=True, train=False, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=64, shuffle=True)

# Create model, loss function, and optimizer
model = torch.nn.Sequential(torch.nn.Linear(INPUT_DIM, HIDDEN_DIM),
                            torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),
                            torch.nn.LogSoftmax(dim=1))
loss_fn = torch.nn.NLLLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


def train():
    """Run an iteration of training and return the loss and accuracy."""
    model.train()
    total_loss = 0
    num_correct = 0
    for data, target in train_loader:
        data = data.view(data.shape[0], -1)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        num_correct += pred.eq(target.data.view_as(pred)).long().sum()
    return total_loss, float(num_correct / len(train_set))


def test():
    """Return the loss and accuracy on the test set."""
    model.eval()
    total_loss = 0
    num_correct = 0
    for data, target in test_loader:
        data = data.view(data.shape[0], -1)
        output = model(data)
        loss = loss_fn(output, target)
        total_loss += loss
        pred = output.data.max(1, keepdim=True)[1]
        num_correct += pred.eq(target.data.view_as(pred)).long().sum()
    return total_loss, float(num_correct / len(test_set))


# Train and test
for epoch in range(NUM_EPOCHS):
    print(f'Epoch {epoch}')
    train_loss, train_acc = train()
    print(f'\tTrain loss: {train_loss}')
    print(f'\tTrain accuracy: {train_acc}')
    test_loss, test_acc = test()
    print(f'\tTest loss: {test_loss}')
    print(f'\tTest accuracy: {test_acc}')
