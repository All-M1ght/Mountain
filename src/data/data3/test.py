import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.data.data3.dataset5 import TallData
import torchvision.models as models
from tensorboardX import SummaryWriter
import time

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# writer = SummaryWriter("/content/drive/My Drive/graph/L1_Resnet_NoDA_Case1_f")

# Hyper parameters
num_epochs = 100  # 20 best 93
num_classes = 1
batch_size = 1
learning_rate = 0.001

# train_dataset = TallData("/content/drive/My Drive/data/data3/case1train.txt", transform=transforms.ToTensor())
test_dataset = TallData("/Users/allmight/PycharmProjects/Mountain/src/data/data3/test.txt", transform=transforms.ToTensor())




# Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


# train_loader_t = torch.utils.data.DataLoader(dataset=train_dataset_t,
#                                            batch_size=100,
#                                            shuffle=True)

# test_loader_t = torch.utils.data.DataLoader(dataset=test_dataset_t,
#                                           batch_size=100,
#                                           shuffle=False)

# train_loader_t = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=771,
#                                            shuffle=True)

# test_loader_t = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=330,
#                                           shuffle=False)


# Convolutional neural network (two convolutional layers)
# class ConvNet(nn.Module):
#     def __init__(self, num_classes=1):
#         super(ConvNet, self).__init__()
#         self.layer1 = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#         self.layer2 = nn.Sequential(
#             nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.layer3 = nn.Sequential(
#             nn.Conv2d(32, 16, kernel_size=5, stride=1, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2))
#
#         self.fc = nn.Linear(28 * 28 * 16, num_classes)
#
#     def forward(self, x):
#         x.to(device)
#         out = self.layer1(x)
#         # print(out.shape)
#         out = self.layer2(out)
#         # print(out.shape)
#         out = self.layer3(out)
#         # print(out.shape)
#         out = out.reshape(out.size(0), -1)
#         # print(out.shape)
#         out = self.fc(out)
#         return out


def predict():
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    trainLoss = 0
    testLoss = 0
    meanLoss_ta = 0
    meanLoss_te = 0
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in train_loader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = model(images)
    #         # print(outputs.shape)
    #         predicted, predictedIndex = torch.max(outputs.data, 1)
    #         correct += torch.abs(predicted - labels).sum().item()
    #         # labels = labels.view(outputs.size(0), 1)
    #         labels = labels.float()
    #         trainLoss = criterion(predicted, labels)
    #         total += labels.size(0)
    #     meanLoss_ta = correct / total
    #     print('Test Average loss of the model on the ' + str(total) + ' train images: {} '.format(meanLoss_ta))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # with torch.no_grad():
            # print(outputs.shape)
            predicted, predictedIndex = torch.max(outputs.data, 1)  # correct = 0
            correct += torch.abs(predicted - labels).sum().item()  # total = 0
            labels = labels.float()
            testLoss = criterion(predicted, labels)
            total += labels.size(0)  # for images, labels in test_loader:
        meanLoss_te = correct / total
        # writer.add_scalar('/Users/allmight/PycharmProjects/Mountain/src/log/test', meanLoss, epoch)
        print('Test Average loss of the model on the ' + str(total) + ' test images: {}'.format(meanLoss_te))
    # writer.add_scalar('L1_Resnet_NoDA_Case1_f/train', meanLoss_ta, epoch)
    # writer.add_scalar('L1_Resnet_NoDA_Case1_f/test', meanLoss_te, epoch)
    # writer.add_scalars('L1_Resnet_NoDA_Case1_f/all', {'trainLoss': meanLoss_ta,
    #                                                   'testLoss': meanLoss_te}, epoch)


# model = ConvNet(num_classes).to(device)
#
# model.load_state_dict(torch.load("newmodel17.ckpt"))

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)

# model = model.to(device)

model.load_state_dict(torch.load("/Users/allmight/PycharmProjects/Mountain/src/data/data3/model/L1_Resnet_NoDA_Case1_f_epoch90.ckpt"))
model = model.to(device)
# model.load_state_dict(torch.load("/Users/allmight/PycharmProjects/Mountain/src/model/L1_Case1_Resnet/epoch29.ckpt"))


# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
with torch.no_grad():
    correct = 0
    total = 0
    real = 0
    pre = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)  # with torch.no_grad():
        # print(outputs.shape)
        predicted, predictedIndex = torch.max(outputs.data, 1)  # correct = 0
        correct += torch.abs(predicted - labels).sum().item()  # total = 0

        labels = labels.float()
        real = labels
        pre = predicted.data.item()
        testLoss = criterion(predicted, labels)
        total += labels.size(0)  # for images, labels in test_loader:
    meanLoss_te = correct / total
    # writer.add_scalar('/Users/allmight/PycharmProjects/Mountain/src/log/test', meanLoss, epoch)
    print("real Altitude = "+str(real)+","+"predict Altitude = "+str(pre)+","+"loss = "+str(meanLoss_te))
    # print('Test Average loss of the model on the ' + str(total) + ' test images: {}'.format(meanLoss_te))
# Train the model
# total_step = len(train_loader)
# for epoch in range(num_epochs):
#     if epoch == 59:
#         learning_rate = learning_rate * 0.1
#     sumLoss = 0
#     for i, (images, labels) in enumerate(train_loader):
#         # print(images.shape)
#         images = images.float()
#         images = images.to(device)
#
#         # print(type(image))
#         labels = labels.to(device)
#
#         # Forward pass
#         # print(images.shape)
#         # print(images.shape)
#
#         # bs, ncrops, c, h, w = images.size()#DA
#         # images = images.view(-1, c, h, w)#DA
#
#         outputs = model(images.to(device))
#         # outputs = outputs.view(bs, ncrops, -1).mean(1)#DA
#
#         # print(type(outputs))
#         # print(type(labels))
#         # format(labels.size, outputs.size)
#         labels = labels.view(outputs.size(0), 1)
#         labels = labels.float()
#         loss = criterion(outputs, labels)
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         if (i + 1) % 1 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
#                   .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
#     # if (i + 1) % 1 == 0:
#     predict()
#     print(time.asctime(time.localtime(time.time())))
#     if (epoch + 1) % 10 == 0:
#         torch.save(model.state_dict(),
#                    '/content/drive/My Drive/model/data3/L1_Resnet_NoDA_Case1_f/epoch' + str(epoch + 1) + '.ckpt')
#
# # writer.export_scalars_to_json("/Users/allmight/PycharmProjects/Mountain/src/logtest.json")
# writer.close()
# from google.colab import drive
# # drive.mount('/content/gdrive')
# # import os
# # # os.chdir("/content/gdrive/My Drive/Colab Notebooks")
# # !ls ""
# !cp "/content/drive/My Drive/dataset2.py" "dataset2.py"
# !pip install tensorboardX

