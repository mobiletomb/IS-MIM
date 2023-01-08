import torch
import random
from monai.transforms import Rotate90
import numpy as np


def rotation(x):
    B, C, Z, H, W = x.shape
    y = []
    label = []
    for i in range(B):
        k = random.randint(0, 3)
        rotate = Rotate90(k, (-2, -1))

        cls = torch.zeros(4, dtype=torch.float32)
        cls[k] = 1
        label.append(cls)
        y.append(rotate(x[i]))

    y = torch.stack(y)

    label = np.stack(label)
    return y, label


if __name__ == '__main__':
    inps = torch.randn((2, 4, 128, 128, 128), dtype=torch.float32)

    x, y = rotation(inps)
    print(x.shape)
    print(y)