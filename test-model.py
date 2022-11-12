from model.GMFupSS import Model
import torch
import os

model = Model()

input_names = ["x.1", ]
output_names = ["output_frame"]
f1 = torch.rand((1, 3, 512, 512))/2
f2 = torch.rand((1, 3, 512, 512))
x = (f1)
print()
print(model(x))
print(x)
