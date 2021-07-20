import torch
import argparse


parser = argparse.ArgumentParser(description='onnx converter')
parser.add_argument('jit_file', type=str, help='input jitted model')
parser.add_argument('onnx_file', type=str, help='output onnx model')
parser.add_argument('--size', default=[1, 2, 44100 * 6], type=int, nargs='+')
parser.add_argument('--input-axes', default=[2], type=int, nargs='+')
parser.add_argument('--output-axes', default=[3], type=int, nargs='+')
args = parser.parse_args()

model = torch.jit.load(args.jit_file)
model.eval()

x = torch.randn(*args.size)
y = model(x)

torch.onnx.export(model,
                  x,
                  args.onnx_file,
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=11,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names=['input'],   # the model's input names
                  output_names=['output'],  # the model's output names
                  dynamic_axes={'input': args.input_axes,    # variable length axes
                                'output': args.output_axes},
                  example_outputs=y)
