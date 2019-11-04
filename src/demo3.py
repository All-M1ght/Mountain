# import src.resnet
import torch
import torch.nn as nn
import urllib
# url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
# # sample execution (requires torchvision)
# from PIL import Image
# from torchvision import transforms
# input_image = Image.open(filename)
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])
# input_tensor = preprocess(input_image)
# input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
#
#
#
# # or any of these variants
#
# model = torch.hub.load('pytorch/vision', 'resnet50', pretrained=True)
#
# model.eval()
#
# # move the input and model to GPU for speed if available
#
# input_batch = input_batch.to(device)
# model.to(device)
#
# with torch.no_grad():
#     output = model(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# print(torch.nn.functional.softmax(output[0], dim=0))
import torchvision.models as models

resnet50 = models.resnet50(pretrained=True)
resnet50.fc = nn.Linear(2048, 2)
