import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .augmentations import DataTransform


class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        self.len = X_train.shape[0]
        if training_mode == "self_supervised" or training_mode == "SupCon":  # no need to apply Augmentations in other modes
            self.aug1, self.aug2 = DataTransform(self.x_data, config)

    def __getitem__(self, index):
        if self.training_mode == "self_supervised" or self.training_mode == "SupCon":
            return self.x_data[index], self.y_data[index], self.aug1[index], self.aug2[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], self.x_data[index]

    def __len__(self):
        return self.len


def data_generator(data_path, configs, training_mode):
    batch_size = configs.batch_size


    if "_1p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_1perc.pt"))
    elif "_5p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_5perc.pt"))
    elif "_10p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_10perc.pt"))
    elif "_50p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_50perc.pt"))
    elif "_75p" in training_mode:
        train_dataset = torch.load(os.path.join(data_path, "train_75perc.pt"))
    elif training_mode == "SupCon":
        train_dataset = torch.load(os.path.join(data_path, "pseudo_train_data.pt"))
    else:
        train_dataset = torch.load(os.path.join(data_path, "train.pt"))

    valid_dataset = torch.load(os.path.join(data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, configs, training_mode)
    valid_dataset = Load_Dataset(valid_dataset, configs, training_mode)
    test_dataset = Load_Dataset(test_dataset, configs, training_mode)

    if train_dataset.__len__() < batch_size:
        batch_size = 16

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size,
                                               shuffle=True, drop_last=configs.drop_last, num_workers=0)
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=batch_size,
                                               shuffle=False, drop_last=configs.drop_last, num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size,
                                              shuffle=False, drop_last=False, num_workers=0)

    return train_loader, valid_loader, test_loader
