import numpy as np
import pandas as pd
import seaborn as sns

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import torchinfo
from torch_snippets import Report
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, KFold
from TrainValidate import TrainValidate, create_weighted_sampler


class PriceDataset(Dataset):
    def __init__(self, features, labels):
        assert len(features) == len(labels)
        self.features = features
        self.labels = labels

    def __getitem__(self, item):
        features = self.features.to_numpy()[item]
        features = torch.tensor(features).float().to(device)

        labels = self.labels.to_numpy()[item]
        price = torch.tensor(labels[1]).float().to(device)

        return features, price

    def __len__(self):
        return len(self.features)

class PredictPrice(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.in_features = in_features

        self.hidden = nn.Sequential(nn.Linear(in_features, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 256),
                                    nn.ReLU(),
                                    nn.Linear(256, 128),
                                    nn.ReLU())
        self.price = nn.Sequential(nn.Linear(128, 1),
                                   nn.ReLU())

    def forward(self, x):
        x = self.hidden(x)
        price = self.price(x)
        return price.squeeze()


if __name__ == '__main__':
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.has_mps:
        device = "mps"

    features = pd.read_csv("../datasets/final_features.csv")
    labels = pd.read_csv("../datasets/final_labels.csv")
    sources = pd.read_csv("../datasets/final_sources.csv")

    scaler = StandardScaler()
    scaler.fit(features)
    features[:] = scaler.transform(features)

    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state=1, test_size=0.1)

    kfold = KFold(n_splits=5, shuffle=True, random_state=1)
    torch.manual_seed(13)
    in_features = len(x_train.iloc[0])
    epochs = 1000

    for fold, (train_id, val_id) in enumerate(kfold.split(x_train.index)):
        train_feature, train_label = x_train.iloc[train_id], y_train.iloc[train_id]
        val_feature, val_label = x_train.iloc[val_id], y_train.iloc[val_id]
        print("\n\n-------------This is fold {}----------------".format(fold))

        train_data = PriceDataset(train_feature, train_label)
        val_data = PriceDataset(val_feature, val_label)

        train_sampler = create_weighted_sampler(train_feature["Sale or Let"].values)
        val_sampler = create_weighted_sampler(val_feature["Sale or Let"].values)

        train_loader = DataLoader(train_data, batch_size=16, drop_last=True)
        val_loader = DataLoader(val_data, batch_size=16, drop_last=True)

        model = PredictPrice(in_features).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        train_validate = TrainValidate(model, nn.MSELoss(), optimizer)
        train_validate.set_loader(train_loader, val_loader)
        train_validate.train(epochs)

        train_validate.save_model("models/all_weighted_sampler_full_epoch_1000_fold_{}.pth".format(fold))
        fig = train_validate.plot_losses()
        fig.savefig("models/all_weighted_sampler_full_epoch_1000_fold_{}.png".format(fold))

