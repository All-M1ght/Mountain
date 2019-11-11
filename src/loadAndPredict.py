import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from src.dataset import MountainDataset
import torchvision.models as models

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 3
num_classes = 2
batch_size = 100
learning_rate = 0.001



train_dataset = MountainDataset("/Users/allmight/PycharmProjects/Mountain/src/data/train.txt", transform=transforms.ToTensor())
test_dataset = MountainDataset("/Users/allmight/PycharmProjects/Mountain/src/data/test.txt", transform=transforms.ToTensor())



# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=1,
                                          shuffle=False)




# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
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
        self.fc = nn.Linear(16 * 16 * 16, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        # print(out.shape)
        out = self.layer2(out)
        # print(out.shape)
        out = self.layer3(out)
        # print(out.shape)
        out = out.reshape(out.size(0), -1)
        # print(out.shape)
        out = self.fc(out)
        return out


# model = ConvNet(num_classes).to(device)
#
#
# model.load_state_dict(torch.load("newmodel20.ckpt"))

model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load("/Users/allmight/PycharmProjects/Mountain/src/model/classfiy/resnetModel.ckpt"))



# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Test Accuracy of the model on the 1635 train images: {} %'.format(100 * correct / total))

with torch.no_grad():
    correct = 0
    total = 0
    i = 0
    index = []
    for images, labels in test_loader:

        # print(labels.shape)
        image = images
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        count = (predicted == labels).sum().item()
        correct += count
        if(count!=1):
            index.append(i)
        i = i + 1
    print('Test Accuracy of the model on the 547 test images: {} %'.format(100 * correct / total))
    for i in index:
        print(test_dataset.imgs[i])

