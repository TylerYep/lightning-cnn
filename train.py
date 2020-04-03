import os
import torch
from torch.nn import functional as F
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl


# class BadModel(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.layer_1 = torch.nn.Linear(28 * 28, 128)
#         self.layer_2 = torch.nn.Linear(128, 256)
#         self.layer_3 = torch.nn.Linear(256, 10)

#     def forward(self, x):
#         batch_size, channels, width, height = x.size()
#         ###
#         # x = x.view(batch_size, -1)
#         ###
#         x = x.view(-1, 1, 56, 56)
#         x = x.permute(1, 0, 3, 2)
#         x = x.reshape((batch_size, -1))
#         ###
#         x = self.layer_1(x)
#         x = torch.relu(x)
#         x = self.layer_2(x)
#         x = torch.relu(x)
#         x = self.layer_3(x)
#         x = torch.log_softmax(x, dim=1)
#         return x


def check_batch_dimension(model, loader, optimizer, test_val=2):
    """
    Verifies that the provided model loads the data correctly. We do this by setting the
    loss to be something trivial (e.g. the sum of all outputs of example i), running the
    backward pass all the way to the input, and ensuring that we only get a non-zero gradient
    on the i-th input.
    See details at http://karpathy.github.io/2019/04/25/recipe/.
    """
    model.eval()
    torch.set_grad_enabled(True)
    data, _ = next(iter(loader))
    optimizer.zero_grad()
    data.requires_grad_()

    output = model(data)
    loss = output[test_val].sum()
    loss.backward()
    print(data.grad)

    error_msg = 'Your model is mixing up data across the batch dimension'
    assert loss != 0
    assert (data.grad[test_val] != 0).any(), error_msg
    assert (data.grad[:test_val] == 0.).all() and (data.grad[test_val+1:] == 0.).all(), error_msg


def load_train_data():
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,))])
    mnist_train = datasets.MNIST('data', train=True, download=False, transform=transform)
    return DataLoader(mnist_train, batch_size=64)


class BadModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = torch.nn.Linear(28 * 28, 128)
        self.layer_2 = torch.nn.Linear(128, 256)
        self.layer_3 = torch.nn.Linear(256, 10)

    def forward(self, x):
        batch_size, channels, width, height = x.size()
        ###
        # x = x.view(batch_size, -1)
        ###
        x = x.view(-1, 1, 56, 56)
        x = x.permute(1, 0, 3, 2)
        x = x.reshape((batch_size, -1))
        ###
        x = self.layer_1(x)
        x = torch.relu(x)
        x = self.layer_2(x)
        x = torch.relu(x)
        x = self.layer_3(x)
        x = torch.log_softmax(x, dim=1)
        return x

    def prepare_data(self):
        datasets.MNIST(os.getcwd(), train=True, download=True)

    def train_dataloader(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.1307,), (0.3081,))])
        mnist_train = datasets.MNIST(os.getcwd(), train=True, download=False, transform=transform)
        return DataLoader(mnist_train, batch_size=64)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = F.nll_loss(logits, y)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}


def main():
    model = BadModel()
    trainer = pl.Trainer()
    trainer.fit(model)

    # loader = load_train_data()
    # model = BadModel()
    # optimizer = optim.Adam(model.parameters(), lr=3e-3)
    # check_batch_dimension(model, loader, optimizer)


if __name__ == "__main__":
    main()
