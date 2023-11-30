from Data_Loaders import Data_Loaders
from Networks import Action_Conditioned_FF

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


def train_model(no_epochs):
    batch_size = 16
    data_loaders = Data_Loaders(batch_size)
    model = Action_Conditioned_FF()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_function = nn.BCELoss()

    losses = []
    min_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
    losses.append(min_loss)

    for epoch_i in range(no_epochs):
        model.train()
        total_loss = 0
        for idx, sample in enumerate(data_loaders.train_loader):
            input, label = sample["input"], sample["label"]
            output = model(input)
            loss = loss_function(
                output, label.unsqueeze(1)
            )  # Ensure labels have correct shape
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()
        average_loss = total_loss / len(data_loaders.train_loader)
        losses.append(average_loss)
        print(f"Epoch {epoch_i+1}/{no_epochs}, Loss: {average_loss}")

        # Evaluate model after each epoch
        test_loss = model.evaluate(model, data_loaders.test_loader, loss_function)
        print(f"Test Loss after Epoch {epoch_i+1}: {test_loss}")

        # Save model if it has the lowest loss
        if test_loss < min_loss:
            min_loss = test_loss
            torch.save(
                model.state_dict(),
                "saved/saved_model.pkl",
                _use_new_zipfile_serialization=False,
            )

    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()


if __name__ == "__main__":
    no_epochs = 30
    train_model(no_epochs)
