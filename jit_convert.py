import torch
import argparse
import json
from jsonschema import validate
import model as module_arch
from utils import get_instance, CONFIG_SCHEMA


parser = argparse.ArgumentParser(description='jit converter')
parser.add_argument('config', type=str, help='config file')
parser.add_argument('checkpoint', type=str, help='training checkpoint')
parser.add_argument('jit_file', type=str, help='output jitted model')
args = parser.parse_args()

config = json.load(open(args.config))
validate(config, schema=CONFIG_SCHEMA)
checkpoint = torch.load(args.checkpoint, map_location='cpu')

model = get_instance(module_arch, config['arch'])
model.load_state_dict(checkpoint['model'])
model.eval()

iteration = checkpoint['trainer']['iteration']
epoch_length = checkpoint['trainer']['epoch_length']
current_epoch = iteration // epoch_length
print(f'trained epochs: {current_epoch}')

jitted = torch.jit.script(model)
print(jitted.code)
jitted.save(args.jit_file)
