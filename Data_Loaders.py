import torch
import torch.utils.data as data
import torch.utils.data.dataset as dataset
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class Nav_Dataset(dataset.Dataset):
    def __init__(self):
        self.data = np.genfromtxt("saved/training_data.csv", delimiter=",")
        # STUDENTS: it may be helpful for the final part to balance the distribution of your collected data
        data_0 = self.data[self.data[:, -1] == 0]
        data_1 = self.data[self.data[:, -1] == 1]
        data_1_rows = data_1.shape[0]
        data_0_rows = min(data_0.shape[0], data_1_rows * 6)
        self.data = np.concatenate((data_1, data_0[0:data_0_rows, :]), axis=0)
        # normalize data and save scaler for inference
        self.scaler = MinMaxScaler()
        self.normalized_data = self.scaler.fit_transform(
            self.data
        )  # fits and transforms
        pickle.dump(
            self.scaler, open("saved/scaler.pkl", "wb")
        )  # save to normalize at inference

    def __len__(self):
        # STUDENTS: __len__() returns the length of the dataset
        return self.data.shape[0]

    def __getitem__(self, idx):
        if not isinstance(idx, int):
            idx = idx.item()
        x = self.normalized_data[idx, 0:-1]
        y = self.normalized_data[idx, -1]
        return {
            "input": torch.tensor(x, dtype=torch.float32),
            "label": torch.tensor(y, dtype=torch.float32),
        }


# STUDENTS: for this example, __getitem__() must return a dict with entries {'input': x, 'label': y}
# x and y should both be of type float32. There are many other ways to do this, but to work with autograding
# please do not deviate from these specifications.


class Data_Loaders:
    def __init__(self, batch_size):
        self.nav_dataset = Nav_Dataset()
        self.batch_size = batch_size
        self.train_loader, self.test_loader = self.get_data_loaders()

    def get_data_loaders(self):
        train_size = int(0.8 * len(self.nav_dataset))
        test_size = len(self.nav_dataset) - train_size
        train_dataset, test_dataset = data.random_split(
            self.nav_dataset, [train_size, test_size]
        )
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = data.DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )
        return train_loader, test_loader


# STUDENTS: randomly split dataset into two data.DataLoaders, self.train_loader and self.test_loader
# make sure your split can handle an arbitrary number of samples in the dataset as this may vary


def main():
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    # STUDENTS : note this is how the dataloaders will be iterated over, and cannot be deviated from
    for idx, sample in enumerate(data_loaders.train_loader):
        _, _ = sample["input"], sample["label"]
    for idx, sample in enumerate(data_loaders.test_loader):
        _, _ = sample["input"], sample["label"]


if __name__ == "__main__":
    main()
