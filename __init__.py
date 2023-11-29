import numpy as np

data = np.loadtxt("submission.csv", delimiter=",")
print(data[:, -1].sum() / 100)
