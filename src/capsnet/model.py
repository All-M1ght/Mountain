########################################
#### Licensed under the MIT license ####
########################################

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import prod
import capsules as caps


class CapsuleNetwork(nn.Module):
    def __init__(self, img_shape, channels, primary_dim, num_classes, out_dim, num_routing, device: torch.device,
                 kernel_size=5):
        super(CapsuleNetwork, self).__init__()
        self.img_shape = img_shape
        self.num_classes = num_classes
        self.device = device

        # self.conv1 = nn.Conv2d(img_shape[0], channels, kernel_size, stride=1, bias=True)
        # self.relu = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        # 28*28*16

        self.primary = caps.PrimaryCapsules(channels, channels, primary_dim, kernel_size)
        # 胶囊数量 =  primary_caps
        # 每个胶囊的维度 = primary_dim
        # primary_caps = int(channels / primary_dim * (img_shape[1] - 2 * (kernel_size - 1)) * (
        #             img_shape[2] - 2 * (kernel_size - 1)) / 4)

        primary_caps = int(channels / primary_dim * (28 - (5 - 1)) * (28 - (5 - 1)))
        # print(img_shape[1])
        # print(img_shape[2])
        # print(kernel_size)
        # print("primary_caps:"+str(primary_caps))
        self.digits = caps.RoutingCapsules(primary_dim, primary_caps, num_classes, out_dim, num_routing,
                                           device=self.device)

        self.decoder = nn.Sequential(
            nn.Linear(out_dim * num_classes, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, int(prod(img_shape))),
            nn.Sigmoid()
        )

    def forward(self, x):
        # out = self.conv1(x)
        # # print(out.shape)
        # out = self.relu(out)
        # print(out.shape)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = self.primary(out)
        # print(out.shape)
        out = self.digits(out)
        # print(out.shape)
        preds = torch.norm(out, dim=-1)

        # Reconstruct the *predicted* image
        _, max_length_idx = preds.max(dim=1)
        # print(max_length_idx)
        y = torch.eye(self.num_classes).to(self.device)
        y = y.index_select(dim=0, index=max_length_idx).unsqueeze(2)
        # print(y)
        # print(preds)
        reconstructions = self.decoder((out * y).view(out.size(0), -1))
        reconstructions = reconstructions.view(-1, *self.img_shape)

        return preds, reconstructions