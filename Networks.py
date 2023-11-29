import torch
import torch.nn as nn


class Action_Conditioned_FF(nn.Module):
    def __init__(self):
        # STUDENTS: __init__() must initiatize nn.Module and define your network's
        # custom architecture
        super(Action_Conditioned_FF, self).__init__()
        self.fc1 = nn.Linear(6, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward(self, input):
        # STUDENTS: forward() must complete a single forward pass through your network
        # and return the output which should be a tensor
        output = self.fc1(input)
        output = nn.functional.relu(output)
        output = self.fc2(output)
        output = nn.functional.relu(output)
        output = self.fc3(output)
        output = nn.functional.relu(output)
        output = self.fc4(output)
        output = nn.functional.sigmoid(output)
        return output

    def evaluate(self, model, test_loader, loss_function):
        # STUDENTS: evaluate() must return the loss (a value, not a tensor) over your testing dataset. Keep in
        # mind that we do not need to keep track of any gradients while evaluating the
        # model. loss_function will be a PyTorch loss function which takes as argument the model's
        # output and the desired output.
        loss = 0
        for idx, sample in enumerate(test_loader):
            input, label = sample["input"], sample["label"]
            output = model(input)
            loss += loss_function(output, label)
        return loss / len(test_loader)


def main():
    model = Action_Conditioned_FF()


if __name__ == "__main__":
    main()
