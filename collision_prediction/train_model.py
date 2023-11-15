from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):

    batch_size =
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()


    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)


    for epoch_i in range(no_epochs):
        model.train()
        for idx, sample in enumerate(data_loaders.train_loader): # sample['input'] and sample['label']
            pass



if __name__ == '__main__':
    no_epochs =
    train_model(no_epochs)
