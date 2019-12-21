import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# primary_caps = 32 * (224 - 2 * (9 - 1)) * (224 - 2 * (9 - 1)) / 4
# print(primary_caps)

# print((224 - 9) / 2 + 1)
#
# zeros = np.zeros((1, 4, 24, 24, 4))
# print(zeros.shape)
# resize = zeros.reshape((1,-1,4))
# print(resize.shape)
# primary_caps = int(16 / 4 * (28 - (5 - 1)) * (28 - (5 - 1)))
# print(primary_caps)

tensor = torch.tensor([False])
t = 5 + tensor
print(t)


