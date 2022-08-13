"""
TrainValidate class in this script is developed based on the two textbooks,
1. Daniel Voigt Godoy. Deep Learning with PyTorch Step-by-Step: A Beginner’s Guide.
2. V Kishore Ayyadevara, Yeshwanth Reddy. Modern Computer Version with Pytorch
"""

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import LambdaLR

from torch_snippets import Report
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
from copy import deepcopy


class TrainValidate:
    def __init__(self, model, loss_fn, optimizer):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer

        self.device = "cpu"
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.has_mps:
            self.device = "mps"
        self.model.to(self.device)

        self.train_loader = None
        self.val_loader = None

        self.log = None
        self.loss = []
        self.val_loss = []
        self.total_epochs = 0

        self.visualization = {}
        self.handles = {}
        self.parameters = {}
        self.gradients = {}

    def set_loader(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train_batch(self, data):
        self.model.train()

        x, y = data
        yhat = self.model(x)

        loss = self.loss_fn(yhat, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def val_batch(self, data):
        self.model.eval()

        x, y = data
        yhat = self.model(x)
        loss = self.loss_fn(yhat, y)

        return loss.item()

    def train(self, epochs, seed=13):
        torch.manual_seed(seed)
        self.log = Report(epochs)

        for epoch in range(epochs):
            self.total_epochs += 1

            N = len(self.train_loader)
            batch_loss = []
            for i, data in enumerate(self.train_loader):
                loss = self.train_batch(data)
                batch_loss.append(loss)
                self.log.record(epoch + (i + 1) / N, train_loss=loss, end='\r')
            self.loss.append(np.mean(batch_loss))

            N = len(self.val_loader)
            batch_val_loss = []
            for i, data in enumerate(self.val_loader):
                loss = self.val_batch(data)
                batch_val_loss.append(loss)
                self.log.record(epoch + (i + 1) / N, val_loss=loss, end='\r')
            self.val_loss.append(np.mean(batch_val_loss))

        self.log.plot()

    def plot_losses(self):
        fig = plt.figure()
        plt.plot(self.loss, label="Training Loss")
        plt.plot(self.val_loss, label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss against epochs")
        return fig

    def save_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = {"model": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "loss": self.loss,
                      "val loss": self.val_loss,
                      "epochs": self.total_epochs}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename="checkpoint.pth"):
        checkpoint = torch.load(filename)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.loss = checkpoint["loss"]
        self.val_loss = checkpoint["val loss"]
        self.total_epochs = checkpoint["epochs"]

    def save_model(self, filename="model.pth"):
        torch.save(self.model, filename)

    @staticmethod
    def make_lr_fn(start_lr: float, end_lr: float, num: int, mode: str = "exp"):
        if mode.lower() == "linear":
            factor = (end_lr / start_lr - 1) / num
            def lr_fn(iter):
                return 1 + iter * factor
        else:
            factor = (np.log(end_lr) - np.log(start_lr)) / num
            def lr_fn(iter):
                return np.exp(factor) ** iter
        return lr_fn

    def test_lr_range(self, data_loader, end, num=100, mode="exp", alpha=0.05):
        previous_state = {"model": deepcopy(self.model.state_dict()),
                          "optimizer": deepcopy(self.optimizer.state_dict())}

        start = self.optimizer.state_dict()["param_groups"][0]["lr"]
        lr_fn = TrainValidate.make_lr_fn(start, end, num, mode)
        scheduler = LambdaLR(self.optimizer, lr_lambda=lr_fn)
        result = {"loss": [], "lr": []}
        iteration = 0

        while iteration < num:
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)

                yhat = self.model(x)
                loss = self.loss_fn(yhat, y)
                loss.backward()

                result["lr"].append(scheduler.get_last_lr()[0])
                if iteration == 0:
                    result["loss"].append(loss.item())
                else:
                    # Exponentially weighted moving average
                    previous_loss = result["loss"][-1]
                    result["loss"].append(alpha * loss.item() + (1 - alpha) * previous_loss)

                iteration += 1
                if iteration == num:
                    break

                self.optimizer.step()
                scheduler.step()
                self.optimizer.zero_grad()

        self.model.load_state_dict(previous_state["model"])
        self.optimizer.load_state_dict(previous_state["optimizer"])

        fig = plt.figure(figsize=(5, 5))
        plt.plot(result["lr"], result["loss"])
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")
        if mode.lower() == "exp":
            plt.xscale("log")
        plt.tight_layout()

        return result, fig


def create_weighted_sampler(data: np.array, bias=None) -> WeightedRandomSampler:
    sale_rent = torch.tensor(data)
    classes, count = sale_rent.unique(return_counts=True)
    weight = 1.0 / count.float()

    if bias is not None:
        assert len(bias) == len(classes)

        for i in range(len(classes)):
            weight[i] += bias[i] * weight[i]

    sample_weight = []
    for p in data:
        for i in range(len(classes)):
            if p == classes[i]:
                sample_weight.append(weight[i].item())

    sample_weight = torch.tensor(sample_weight)

    generator = torch.Generator()
    sampler = WeightedRandomSampler(weights=sample_weight,
                                    num_samples=len(data),
                                    generator=generator,
                                    replacement=True)

    return sampler


if __name__ == '__main__':
    data = np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 1])
    create_weighted_sampler(data, [0.1, 0])
