import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
from matplotlib import pyplot as plt


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

        self.train_loader = None
        self.val_loader = None

        self.losses = []
        self.val_losses = []
        self.total_epochs = 0

        self.train_step = self._make_train_step()
        self.val_step = self._make_val_step()

    def set_loaders(self, train_loader, val_loader=None):
        self.train_loader = train_loader
        self.val_loader = val_loader

    def _make_train_step(self):
        def perform_train_step(x, y):
            self.model.train()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            return loss.item()
        return perform_train_step

    def _make_val_step(self):
        def perform_val_step(x, y):
            self.model.eval()

            yhat = self.model(x)
            loss = self.loss_fn(yhat, y)
            return loss.item()
        return perform_val_step

    def _mini_batch(self, validation=False):
        if validation:
            data_loader = self.val_loader
            step = self.val_step
        else:
            data_loader = self.train_loader
            step = self.train_step

        if data_loader is None:
            return None

        mini_batch_losses = []
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            mini_batch_losses.append(step(x_batch, y_batch))

        return np.mean(mini_batch_losses)

    def train(self, n_epochs, seed=13):
        torch.manual_seed(seed)

        for epoch in range(n_epochs):
            self.total_epochs += 1

            self.losses.append(self._mini_batch())
            with torch.no_grad():
                self.val_losses.append(self._mini_batch(validation=True))

    @torch.no_grad()
    def predict(self, x):
        self.model.eval()

        x = torch.as_tensor(x).float().to(self.device)
        yhat = self.model(x)

        return yhat.detach().cpu().numpy()

    def save(self, filename):
        checkpoint = {"epoch": self.total_epochs,
                      "model": self.model.state_dict(),
                      "optimizer": self.optimizer.state_dict(),
                      "train loss": self.losses,
                      "val loss": self.val_losses}
        torch.save(checkpoint, filename)

    def load(self, filename):
        checkpoint = torch.load(filename)

        self.total_epochs = checkpoint["epoch"]
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.losses = checkpoint["train loss"]
        self.val_losses = checkpoint["val loss"]

    def plot_losses(self):
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.losses, label="Training Loss")

        if self.val_loader:
            plt.plot(self.val_losses, label="Validation Loss")

        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()

        return fig
