from model.GMFupSS import Model
import torch
import os

model = Model()

input_names = ["x.1", ]
output_names = ["output_frame"]
f1 = torch.rand((1, 3, 512, 512))
f2 = torch.rand((1, 3, 512, 512))
x = (f1)
torch.onnx.export(model,               # model being run
                  x,                         # model input (or a tuple for multiple inputs)
                  "gmfss.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=16,          # the ONNX version to export the model to
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'],
                  verbose=False)
