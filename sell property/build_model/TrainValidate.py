import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from torch_snippets import Report
from sklearn.model_selection import KFold
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

        self.log = None
        self.loss = []
        self.val_loss = []
        self.total_epochs = 0

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
