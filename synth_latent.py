import torch
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
from jsonschema import validate
import model as module_arch
from utils import get_instance, CONFIG_SCHEMA


parser = argparse.ArgumentParser(description='latent visualizer')
parser.add_argument('config', type=str, help='config file')
parser.add_argument('checkpoint', type=str, help='training checkpoint')
parser.add_argument('--log_dir', type=str,
                    default='./visualize', help='tb logs')
parser.add_argument('--points', type=int, default=1000)
args = parser.parse_args()

config = json.load(open(args.config))
validate(config, schema=CONFIG_SCHEMA)
checkpoint = torch.load(args.checkpoint, map_location='cpu')


model = get_instance(module_arch, config['arch'])
model.load_state_dict(checkpoint['model'])
model.eval()

k_mu = model.cluster_mu
k_logvar = model.cluster_logvar
print(k_mu)

writer = SummaryWriter(log_dir=args.log_dir)

latent_z = []
for i in range(5):
    latent_z.append(
        k_mu[i] + torch.randn(args.points, k_mu.shape[1]) *
        torch.exp(0.5 * k_logvar[i])
    )

latent_z = torch.cat(latent_z, 0)
print(latent_z.shape)
latent_z = latent_z.view(-1, latent_z.shape[-1])
metadata = ['drums'] * args.points + ['bass'] * args.points + \
    ['other'] * args.points + ['vocals'] * \
    args.points + ['silence'] * args.points

writer.add_embedding(latent_z, metadata=metadata)
writer.close()
