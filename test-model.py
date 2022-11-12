from model.GMFupSS import Model
import torch
import os
from torchvision import transforms

from PIL import Image
model = Model()

img1 = Image.open("demo/stacked.jpg")
img1 = img1.resize((128,128))
convert_tensor = transforms.ToTensor()
import cv2
img1=convert_tensor(img1).unsqueeze(dim=0)

input_names = ["x.1", ]
output_names = ["output_frame"]
f1 = torch.rand((1, 3, 512, 512))/2
f2 = torch.rand((1, 3, 512, 512))
x = (img1)
print(img1.shape)
print("finished")
out=x
print("model")
out = model(x)
print("data")
print(x)
out=out.squeeze(dim=0)
out=out.detach().cpu().numpy().transpose(1, 2, 0)
print(out.shape)
print(out.shape)

print(out.shape)

print(out.shape)

cv2.imwrite("out.png",out*255)