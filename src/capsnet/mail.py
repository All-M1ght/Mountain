import os
import torch
import torchvision
import torchvision.transforms as transforms
from trainer import CapsNetTrainer
from dataset import TallData
import argparse
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ['1', '2', '3', '4','5', '6', '7', '8', '9']
# transform = transforms.Compose([
#     # shift by 2 pixels in either direction with zero padding.
#     transforms.RandomCrop(size, padding=2),
#     transforms.ToTensor(),
#     transforms.Normalize( mean, std )
# ])
loaders = {}
# trainset = datasets['MNIST'](root="/content", train=True, download=True, transform=transform)
train_dataset = TallData("/content/drive/My Drive/data/data3/case1train_caps.txt", transform=transforms.Compose([
    # transforms.RandomCrop(224),
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))
loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# loaders['train'] = train_loader

# testset = datasets['MNIST'](root="/content", train=False, download=True, transform=transform)
test_dataset = TallData("/content/drive/My Drive/data/data3/case1test_caps.txt", transform=transforms.Compose([
    # transforms.RandomCrop(224),
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
]))

loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# Run
caps_net = CapsNetTrainer(loaders=loaders,batch_size=1,learning_rate=0.001,num_routing=3,lr_decay=0.96)
caps_net.run(100, classes=classes)

#!cp "/content/drive/My Drive/capsnet/trainer.py" "trainer.py"
#!cp "/content/drive/My Drive/capsnet/loss.py" "loss.py"
#!cp "/content/drive/My Drive/capsnet/model.py" "model.py"
#!cp "/content/drive/My Drive/capsnet/capsules.py" "capsules.py"
# !cp "/content/drive/My Drive/capsnet/dataset.py" "dataset2.py"

