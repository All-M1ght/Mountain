import torch
import torch.nn as nn
import torchvision.transforms as transforms
from src.dataset2 import TallData
import torchvision.models as models
from tensorboardX import SummaryWriter

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
# Hyper parameters
num_epochs = 10  #20 best 93
num_classes = 1
batch_size = 2
learning_rate = 0.001







train_dataset = TallData("/Users/allmight/PycharmProjects/Mountain/src/data/tt", transform=transforms.ToTensor())
test_dataset = TallData("/Users/allmight/PycharmProjects/Mountain/src/data/regression/tt", transform=transforms.ToTensor())





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


def predict():
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    trainLoss = 0
    testLoss = 0
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device)
            # print(type(images))
            labels = labels.to(device)
            outputs = model(images)
            predicted, predictedIndex = torch.max(outputs.data, 1)
            correct += torch.abs(predicted - labels).sum().item()
            # labels = labels.view(outputs.size(0), 1)
            labels = labels.float()
            trainLoss = criterion(predicted,labels)
            total += labels.size(0)
        meanLoss = correct / total
        print(trainLoss.item())
        print(meanLoss)
        print('Test Average loss of the model on the ' + str(total) + ' train images: {} '.format(trainLoss.item()))

    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)  # with torch.no_grad():
            predicted, predictedIndex = torch.max(outputs.data, 1)  # correct = 0
            correct += torch.abs(predicted - labels).sum().item()  # total = 0
            labels = labels.float()
            testLoss = criterion(predicted, labels)
            total += labels.size(0)  # for images, labels in test_loader:
        meanLoss = correct / total
        # writer.add_scalar('/Users/allmight/PycharmProjects/Mountain/src/log/test', meanLoss, epoch)
        # print('Test Average loss of the model on the ' + str(total) + ' test images: {}'.format(meanLoss))
    writer.add_scalar('data/train', trainLoss, epoch)
    writer.add_scalar('data/test', testLoss, epoch)
    writer.add_scalars('data/all', {'trainLoss': trainLoss,
                                             'testLoss': testLoss}, epoch)

# model = ConvNet(num_classes).to(device)
#
# model.load_state_dict(torch.load("newmodel17.ckpt"))

model = models.resnet50(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)
#
# model.load_state_dict(torch.load("resnet50-5.ckpt"))
# model.load_state_dict(torch.load("/Users/allmight/PycharmProjects/Mountain/src/model/L1_Case1_Resnet/epoch29.ckpt"))



# Loss and optimizer
# criterion = nn.CrossEntropyLoss()
# criterion = nn.MSELoss()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
total_step = len(train_loader)
for epoch in range(num_epochs):
    sumLoss = 0
    for i, (images, labels) in enumerate(train_loader):
        # print(images.shape)
        images = images.to(device)
        labels = labels.to(device)
        # Forward pass
        # print(images.shape)
        # print(images.shape)
        outputs = model(images)

        # print(type(outputs))
        # print(type(labels))
        # format(labels.size, outputs.size)
        labels = labels.view(outputs.size(0), 1)
        labels = labels.float()
        loss = criterion(outputs, labels)
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    # torch.save(model.state_dict(), '/Users/allmight/PycharmProjects/Mountain/src/model/L1_Case1_Resnet_dA/epoch'+str(epoch)+'.ckpt')
    predict()

writer.export_scalars_to_json("/Users/allmight/PycharmProjects/Mountain/src/logtest.json")
writer.close()

# Test the model
# model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in train_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         predicted, predictedIndex = torch.max(outputs.data, 1)
#         correct += torch.abs(predicted - labels).sum().item()
#         total += labels.size(0)
#     print('Test Average loss of the model on the '+str(total)+' train images: {} '.format(correct / total))
#
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)# with torch.no_grad():
#         predicted, predictedIndex = torch.max(outputs.data, 1)#     correct = 0
#         correct += torch.abs(predicted - labels).sum().item()#     total = 0
#         total += labels.size(0)#     for images, labels in test_loader:
#     print('Test Average loss of the model on the '+str(total)+' test images: {}'.format(correct / total))










