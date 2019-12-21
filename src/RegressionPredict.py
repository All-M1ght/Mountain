import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.dataset2 import TallData
import torchvision.models as models

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 1  #20 best 93
num_classes = 1
batch_size = 100
learning_rate = 0.01



train_dataset = TallData("/Users/allmight/PycharmProjects/Mountain/src/data/tt.txt", transform=transforms.ToTensor())
test_dataset = TallData("/Users/allmight/PycharmProjects/Mountain/src/data/regression/case1test", transform=transforms.ToTensor())


# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Convolutional neural network (two convolutional layers)
class ConvNet(nn.Module):
    def __init__(self, num_classes=1):
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



        self.fc = nn.Linear(28 * 28 * 16, num_classes)

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
# model.load_state_dict(torch.load("newmodel17.ckpt"))


model = models.resnet50(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
#

model.load_state_dict(torch.load("/Users/allmight/PycharmProjects/Mountain/src/model/L1_Case1_Resnet_dA/step52epoch2.ckpt"))






# Test the model
model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        predicted, predictedIndex = torch.max(outputs.data, 1)
        # print(predicted.data.item())
        correct += torch.abs(predicted - labels).sum().item()
        # print(correct)
        total += labels.size(0)
    print('Test Average loss of the model on the '+str(total)+' train images: {} '.format(correct / total))

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         predicted, predictedIndex = torch.max(outputs.data, 1)
#         correct += torch.abs(predicted - labels).sum().item()
#         total += labels.size(0)
#     print('Test Average loss of the model on the '+str(total)+' test images: {}'.format(correct / total))

# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#
#     print('Test Accuracy of the model on the 547 test images: {} %'.format(100 * correct / total))
#
# # Save the model checkpoint
# torch.save(model.state_dict(), 'resnet50-8.ckpt')