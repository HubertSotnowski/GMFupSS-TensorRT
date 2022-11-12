import onnx
model = onnx.load('gmfss.onnx')
output = model.graph.output

input_all = model.graph.input
print(input_all)