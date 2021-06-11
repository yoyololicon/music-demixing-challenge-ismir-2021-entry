import random
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.cuda import amp
from torchvision.transforms import Compose
import argparse
import json
from datetime import datetime
import os
from jsonschema import validate
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan, Checkpoint
from ignite.contrib.handlers.tensorboard_logger import *
import torch_optimizer

import dataset as module_data
import loss as module_loss
import model as module_arch
import transform as module_transform
from sync_batchnorm import convert_model

from utils import get_instance, CONFIG_SCHEMA, MWF


parser = argparse.ArgumentParser(description='SS Trainer')

parser.add_argument('config', type=str, help='config file')
parser.add_argument('--checkpoint', type=str, default=None,
                    help='training checkpoint')
parser.add_argument('--weights', type=str, default=None,
                    help='initial model weights')

args = parser.parse_args()

config = json.load(open(args.config))
validate(config, schema=CONFIG_SCHEMA)

if torch.cuda.is_available():
    device = 'cuda:1'
    device_ids = [1,2,3,4,5,7]
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# define data augmentation transformss
cpu_trsfm = Compose([
    module_transform.RandomGain(),
    module_transform.RandomSwapLR()
])

# your device transforms needs to handle with batches
device_trsfm = nn.Sequential().to(device)

train_data = get_instance(
    module_data, config['dataset']['train'], transform=cpu_trsfm)
val_data = get_instance(module_data, config['dataset']['valid'])

train_loader = DataLoader(train_data, **config['data_loader']['train'])
val_loader = DataLoader(val_data, **config['data_loader']['valid'])


class MyDataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

model = get_instance(module_arch, config['arch']).to(device)

if device_ids:
    model = MyDataParallel(model, device_ids=device_ids)
    model = convert_model(model).to(device)
try:
    optimizer = get_instance(optim, config['optimizer'], model.parameters())
except AttributeError:
    optimizer = get_instance(
        torch_optimizer, config['optimizer'], model.parameters())

scheduler = get_instance(optim.lr_scheduler, config['lr_scheduler'], optimizer)

criterion = get_instance(module_loss, config['loss']).to(device)

sdr = module_loss.SDR()
mwf_kwargs = config.get('MWF', {})
mwf = MWF(**mwf_kwargs)

print('Trainable parameters: {}'.format(sum(p.numel()
                                            for p in model.parameters() if p.requires_grad)))


model_name = config['name']
log_dir = config['trainer']['log_dir']
save_dir = config['trainer']['save_dir']
targets = config['trainer']['targets']
amp_enabled = config['trainer']['amp_enabled']
n_fft = config['trainer']['n_fft']
hop_length = config['trainer']['hop_length']
accumulation_steps = config['trainer']['cum_steps']
extra_monitor = config['trainer']['extra_monitor']
validate_every = config['trainer']['validate_every']
val_epoch_length = config['trainer']['val_epoch_length']
patience = config['trainer']['patience']
epochs = config['trainer']['epochs']

# get target index
targets_idx = []
for t in targets:
    targets_idx.append(train_data.sources.index(t))
assert len(targets_idx) > 0
targets_idx = sorted(targets_idx)


scaler = amp.GradScaler(enabled=amp_enabled)
spec = module_arch.Spec(n_fft, hop_length).to(device)


def process_function(engine, batch):
    model.train()

    x, y = batch
    x, y = x.to(device), y.to(device)
    if len(device_trsfm):
        y = device_trsfm(y)
        x = y.sum(1)
    y = y[:, targets_idx].squeeze(1)

    X = spec(x)
    Y = spec(y)
    X_mag = X.abs()
    with amp.autocast(enabled=amp_enabled):
        pred_mask = model(X_mag)

    loss, extra_losses = criterion(pred_mask, Y, X, y, x)
    loss /= accumulation_steps
    scaler.scale(loss).backward()

    if engine.state.iteration % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    result = {'loss': loss.item() * accumulation_steps}
    result.update(extra_losses)
    return result


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        y = y[:, targets_idx].squeeze(1)
        x, y = x.to(device), y.to(device)

        X = spec(x)
        Y = spec(y)
        X_mag = X.abs()
        with amp.autocast(enabled=amp_enabled):
            pred_mask = model(X_mag)

        loss, extra_losses = criterion(pred_mask, Y, X, y, x)
        result = {'loss': loss.item()}
        result.update(extra_losses)

        if pred_mask.ndim == X_mag.ndim:
            pred_mask = pred_mask.unsqueeze(1)

        Y = mwf(pred_mask, X)
        xpred = spec(Y, inverse=True)

        batch = xpred.shape[0]
        sdrs = sdr(
            xpred.view(-1, *xpred.shape[-2:]), y.view(-1, *y.shape[-2:])).view(batch, -1).mean(0)

        for i, t in enumerate(targets_idx):
            result[f'{val_data.sources[t]}_sdr'] = sdrs[i].item()
        result['avg_sdr'] = sdrs.mean().item()
        return result


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)

RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
Average(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
for k in extra_monitor:
    RunningAverage(output_transform=lambda x, m=k: x[m]).attach(trainer, k)
    Average(output_transform=lambda x, m=k: x[m]).attach(evaluator, k)
for t in targets_idx:
    k = f'{val_data.sources[t]}_sdr'
    Average(output_transform=lambda x, m=k: x[m]).attach(evaluator, k)
Average(output_transform=lambda x: x['avg_sdr']).attach(evaluator, 'avg_sdr')


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    output_str = f'Trainer Results - Epoch {engine.state.epoch} - Avg loss: {avg_loss:.2f}'
    for k in extra_monitor:
        output_str += f' Avg {k}: {engine.state.metrics[k]:.2f}'
    print(output_str)


def print_logs(engine, dataloader):
    evaluator.run(dataloader, max_epochs=1, epoch_length=val_epoch_length)
    metrics = evaluator.state.metrics
    avg_loss = metrics['loss']
    avg_sdr = metrics['avg_sdr']
    scheduler.step(avg_loss)

    output_str = f'Evaluater Results - Epoch {engine.state.epoch} - Avg loss: {avg_loss:.2f}'
    for k in extra_monitor:
        output_str += f' Avg {k}: {metrics[k]:.2f}'

    output_str += f' Avg SDR: {avg_sdr:.2f}'

    print(output_str)


trainer.add_event_handler(Events.EPOCH_COMPLETED(
    every=validate_every), print_logs, val_loader)

# Tqdm
pbar = ProgressBar(persist=True)
pbar.attach(trainer, 'all')

# Create a logger
start_time = datetime.now().strftime('%m%d_%H%M%S')
tb_logger = TensorboardLogger(
    log_dir=os.path.join(log_dir, model_name, start_time))
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    metric_names='all'
)
tb_logger.attach_output_handler(
    evaluator,
    event_name=Events.EPOCH_COMPLETED,
    tag="validation",
    metric_names='all',
    global_step_transform=global_step_from_engine(trainer),
)
tb_logger.attach(
    trainer,
    event_name=Events.EPOCH_COMPLETED,
    log_handler=WeightsHistHandler(model)
)
tb_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_STARTED,
    optimizer=optimizer,
)

# add model graph
test_input = spec(torch.from_numpy(
    val_data[0][0]).to(device)).abs().unsqueeze(0)
# tb_logger.writer.add_graph(model, input_to_model=test_input)
tb_logger.writer.add_graph(model.module, input_to_model=test_input)

# early stop


def score_function(engine):
    return -engine.state.metrics['loss']


def sdr_score_function(engine):
    return engine.state.metrics['avg_sdr']


handler = EarlyStopping(
    patience=patience, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

checkpointer = ModelCheckpoint(
    save_dir, model_name, score_function=sdr_score_function, n_saved=4, create_dir=True, require_empty=False)
to_save = {
    'model': model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'trainer': trainer,
    'scaler': scaler
}
evaluator.add_event_handler(
    Events.COMPLETED,
    checkpointer,
    to_save
)


def predict_samples(engine):
    model.eval()
    with torch.no_grad():
        x, _ = val_data[random.randrange(len(val_data))]
        x = torch.from_numpy(x)
        tb_logger.writer.add_audio('mixture', x.t(), engine.state.epoch)

        X = spec(x.to(device)).unsqueeze(0)
        X_mag = X.abs()
        with amp.autocast(enabled=amp_enabled):
            pred_mask = model(X_mag)

        if pred_mask.ndim == X_mag.ndim:
            pred_mask = pred_mask.unsqueeze(1)

        Y = mwf(pred_mask, X).squeeze()

        xpred = spec(Y, inverse=True).cpu().clip(-1, 1)

        if len(targets_idx) > 1:
            xpred = xpred.transpose(1, 2)
            for i, t in enumerate(targets_idx):
                tb_logger.writer.add_audio(
                    val_data.sources[t], xpred[i], engine.state.epoch)
        else:
            xpred = xpred.t()
            tb_logger.writer.add_audio(
                val_data.sources[targets_idx[0]], xpred, engine.state.epoch)


trainer.add_event_handler(Events.EPOCH_COMPLETED, predict_samples)

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

if args.weights:
    model.load_state_dict(torch.load(args.weights, map_location=device))

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

e = trainer.run(train_loader, max_epochs=epochs)

tb_logger.close()
