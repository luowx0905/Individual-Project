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

    def attach_hooks(self, layer_to_hook, hook_fn=None):
        self.visualization = {}

        modules = list(self.model.named_modules())
        layer_names = {layer: name for layer, name in modules[1:]}

        if hook_fn is None:
            def hook_fn(layer, inputs, outputs):
                name = layer_names[layer]
                values = outputs.detach().cpu().numpy()

                if self.visualization[name] is None:
                    self.visualization[name] = values
                else:
                    self.visualization[name] = np.concatenate([self.visualization[name], values])

        for name, layer in modules:
            if name in layer_to_hook:
                self.visualization[name] = None
                self.handles[name] = layer.register_forward_hook(hook_fn)

    def remove_hook(self):
        for handle in self.handles.values():
            handle.remove()
        self.handles = {}

    def get_parameters(self, layer_to_hook):
        if not isinstance(layer_to_hook, list):
            layer_to_hook = [layer_to_hook]

        modules = list(self.model.named_modules())
        layer_names = {layer: name for name, layer in modules}

        self.parameters = {}

        for name, layer in modules:
            if name in layer_to_hook:
                self.parameters.update({name: {}})
                for parm_id, p in layer.named_parameters():
                    self.parameters[name].update({parm_id: []})

        def hook_fn(layer, inputs, outputs):
            name = layer_names[layer]
            for parm_id, parameter in layer.names_parameters():
                self.parameters[name][parm_id].append(parameter.tolist())

        self.attach_hooks(layer_to_hook, hook_fn)

    def get_gradient(self, layer_to_hook):
        if not isinstance(layer_to_hook, list):
            layer_to_hook = [layer_to_hook]

        modules = list(self.model.named_modules())
        self.gradients = {}

        def make_log_fn(name, parm_id):
            def log_fn(grad):
                self.gradients[name][parm_id].append(grad.tolist())
            return log_fn

        for name, layer in self.model.named_modules():
            if name in layer_to_hook:
                self.gradients.update({name: {}})
                for parm_id, p in layer.names_parameters():
                    if p.requires_grad:
                        self.gradients[name].update({parm_id: []})
                        log_fn = make_log_fn(name, parm_id)
                        self.handles[f"{name}.{parm_id}.grad"] = p.register_hook(log_fn)


def create_weighted_sampler(data: np.array) -> WeightedRandomSampler:
    sale_rent = torch.tensor(data)
    classes, count = sale_rent.unique(return_counts=True)
    weight = 1.0 / count.float()

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
    start = 0.01
    end = 0.1
    num = 10
    lr_fn = TrainValidate.make_lr_fn(start, end, num, mode="exp")
    print((start * lr_fn(np.arange(num + 1))))
