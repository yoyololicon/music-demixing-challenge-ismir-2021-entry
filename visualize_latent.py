import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from tqdm import tqdm
import random
from jsonschema import validate
import model as module_arch
import dataset as module_data
from utils import get_instance, CONFIG_SCHEMA


parser = argparse.ArgumentParser(description='latent visualizer')
parser.add_argument('config', type=str, help='config file')
parser.add_argument('checkpoint', type=str, help='training checkpoint')
parser.add_argument('--log_dir', type=str,
                    default='./visualize', help='tb logs')
parser.add_argument('--steps', type=int, default=20)
args = parser.parse_args()

config = json.load(open(args.config))
validate(config, schema=CONFIG_SCHEMA)
checkpoint = torch.load(args.checkpoint, map_location='cpu')

if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'


model = get_instance(module_arch, config['arch'])
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

val_data = get_instance(module_data, config['dataset']['valid'])
n_fft = config['trainer'].get('n_fft', 4096)
hop_length = config['trainer'].get('hop_length', 1024)
spec = module_arch.Spec(n_fft, hop_length).to(device)

writer = SummaryWriter(log_dir=args.log_dir)


if args.steps < len(val_data):
    indexes = random.sample(range(len(val_data)), args.steps)
else:
    indexes = list(range(len(val_data)))


with torch.no_grad():
    latent_z = []
    for i in tqdm(indexes):
        _, y = val_data[i]
        y = torch.from_numpy(y).to(device).unsqueeze(0)
        Y = spec(y)
        recon, z, *_ = model(Y.abs())
        latent_z.append(z.cpu())
latent_z = torch.cat(latent_z, 0).transpose(0, 1)

latent_z = latent_z.reshape(4, -1, latent_z.shape[-1])


num_samples = latent_z.shape[1]
print(latent_z.shape)
latent_z = latent_z.view(-1, latent_z.shape[-1])
metadata = ['drums'] * num_samples + ['bass'] * num_samples + \
    ['other'] * num_samples + ['vocals'] * num_samples

writer.add_embedding(latent_z, metadata=metadata)
writer.close()
