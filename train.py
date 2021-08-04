import random
import torch
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.cuda import amp
from torchvision.transforms import Compose
from torchinfo import summary
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
parser.add_argument('-d', '--device', default=0,
                    type=int, help='cuda device number')
parser.add_argument('--device_ids',  default=None, type=int, nargs='+',
                    help='indices of GPUs for DataParallel (default: None)')

args = parser.parse_args()

config = json.load(open(args.config))
validate(config, schema=CONFIG_SCHEMA)

if torch.cuda.is_available():
    device = f"cuda:{args.device}"
    device_ids = args.device_ids
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# define data augmentation transformss
cpu_trsfm = Compose([
    module_transform.RandomGain(),
    module_transform.RandomSwapLR(),
    module_transform.RandomFlipPhase()
])

# your device transforms needs to handle with batches
device_trsfm = nn.Sequential(
    module_transform.RandomPitch(),
    module_transform.SpeedPerturb()
).to(device)

train_data = get_instance(
    module_data, config['dataset']['train'], transform=cpu_trsfm)
val_data = get_instance(module_data, config['dataset']['valid'])

train_loader = DataLoader(train_data, **config['data_loader']['train'])
val_loader = DataLoader(val_data, **config['data_loader']['valid'])


model = get_instance(module_arch, config['arch']).to(device)

if device_ids:
    print(f'using multi-GPU')
    model = nn.DataParallel(model, device_ids=device_ids)
    model = convert_model(model)

try:
    optimizer = get_instance(optim, config['optimizer'], model.parameters())
except AttributeError:
    optimizer = get_instance(
        torch_optimizer, config['optimizer'], model.parameters())

scheduler = get_instance(optim.lr_scheduler, config['lr_scheduler'], optimizer)

criterion = get_instance(module_loss, config['loss']).to(device)

sdr = module_loss.SDR()


model_name = config['name']
log_dir = config['trainer']['log_dir']
save_dir = config['trainer']['save_dir']
targets = config['trainer']['targets']
amp_enabled = config['trainer']['amp_enabled']
accumulation_steps = config['trainer']['cum_steps']
extra_monitor = config['trainer']['extra_monitor']
validate_every = config['trainer']['validate_every']
val_epoch_length = config['trainer']['val_epoch_length']
patience = config['trainer']['patience']
epochs = config['trainer']['epochs']
n_fft = config['trainer'].get('n_fft', 4096)
hop_length = config['trainer'].get('hop_length', 1024)
time_domain = config['trainer'].get('time_domain', False)
mwf_kwargs = config.get('MWF', {})

if time_domain:
    def spec(x): return x
    def mwf(x): return x
else:
    spec = module_arch.Spec(n_fft, hop_length).to(device)
    mwf = MWF(**mwf_kwargs)

# get target index
targets_idx = []
for t in targets:
    targets_idx.append(train_data.sources.index(t))
assert len(targets_idx) > 0
targets_idx = sorted(targets_idx)

scaler = amp.GradScaler(enabled=amp_enabled)


def _process_core(x, y):
    if time_domain:
        with amp.autocast(enabled=amp_enabled):
            pred = model(x)
        return criterion(pred, y, x) + (pred,)

    X = spec(x)
    Y = spec(y)
    X_mag = X.abs()
    with amp.autocast(enabled=amp_enabled):
        pred_mask = model(X_mag)
    return criterion(pred_mask, Y, X, y, x) + (pred_mask, X)


def process_function(engine, batch):
    model.train()

    x, y = batch
    x, y = x.to(device), y.to(device)
    if len(device_trsfm):
        y = device_trsfm(y)
        x = y.sum(1)
    y = y[:, targets_idx].squeeze(1)

    loss, extra_losses, *_ = _process_core(x, y)
    loss /= accumulation_steps
    scaler.scale(loss).backward()

    if engine.state.iteration % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    result = {'loss': loss.item() * accumulation_steps}
    result.update(extra_losses)
    return result


def _eval_core(x, y):
    ret = _process_core(x, y)
    if time_domain:
        return ret

    loss, extra_losses, pred_mask, X = ret

    if pred_mask.ndim == X.ndim:
        pred_mask = pred_mask.unsqueeze(1)

    Y = mwf(pred_mask, X)
    xpred = spec(Y, inverse=True)
    return loss, extra_losses, xpred


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        y = y[:, targets_idx].squeeze(1)
        x, y = x.to(device), y.to(device)

        loss, extra_losses, xpred = _eval_core(x, y)
        result = {'loss': loss.item()}
        result.update(extra_losses)

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
    event_name=Events.EPOCH_COMPLETED(every=validate_every),
    log_handler=WeightsHistHandler(model)
)
tb_logger.attach_opt_params_handler(
    trainer,
    event_name=Events.ITERATION_STARTED,
    optimizer=optimizer,
)

# add model graph
# use torchinfo
test_input = torch.from_numpy(val_data[0][0]).to(device).unsqueeze(0)
if not time_domain:
    test_input = spec(test_input).abs()
summary(model,
        input_data=test_input,
        device=device,
        col_names=("input_size", "output_size", "num_params", "kernel_size",
                   "mult_adds"),
        col_width=16,
        row_settings=("depth", "var_names"))
# tb_logger.writer.add_graph(model, input_to_model=test_input)

# early stop


def score_function(engine):
    return -engine.state.metrics['loss']


def sdr_score_function(engine):
    return engine.state.metrics['avg_sdr']


handler = EarlyStopping(
    patience=patience, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

checkpointer = ModelCheckpoint(
    save_dir, model_name, score_function=sdr_score_function, n_saved=2, create_dir=True, require_empty=False)
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


def _predict_core(x):
    x = x.unsqueeze(0)
    if time_domain:
        with amp.autocast(enabled=amp_enabled):
            return model(x).squeeze()

    X = spec(x)
    X_mag = X.abs()
    with amp.autocast(enabled=amp_enabled):
        pred_mask = model(X_mag)

    if pred_mask.ndim == X_mag.ndim:
        pred_mask = pred_mask.unsqueeze(1)

    Y = mwf(pred_mask, X).squeeze()
    xpred = spec(Y, inverse=True)
    return xpred


def predict_samples(engine):
    model.eval()
    with torch.no_grad():
        x, _ = val_data[random.randrange(len(val_data))]
        x = torch.from_numpy(x)
        tb_logger.writer.add_audio('mixture', x.t(), engine.state.epoch)

        xpred = _predict_core(x.to(device)).cpu().float().clip(-1, 1)

        if len(targets_idx) > 1:
            xpred = xpred.transpose(1, 2)
            for i, t in enumerate(targets_idx):
                tb_logger.writer.add_audio(
                    val_data.sources[t], xpred[i], engine.state.epoch)
        else:
            xpred = xpred.t()
            tb_logger.writer.add_audio(
                val_data.sources[targets_idx[0]], xpred, engine.state.epoch)


trainer.add_event_handler(Events.EPOCH_COMPLETED(
    every=validate_every), predict_samples)


if args.checkpoint:
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    iteration = checkpoint['trainer']['iteration']
    epoch_length = checkpoint['trainer']['epoch_length']
    current_epoch = iteration // epoch_length
    epoch_length = len(train_loader)
    iteration = current_epoch * epoch_length
    checkpoint['trainer']['iteration'] = iteration
    checkpoint['trainer']['epoch_length'] = epoch_length
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)

if args.weights:
    weights = torch.load(args.weights, map_location='cpu')
    try:
        weights = weights['model']
    except:
        pass
    model.load_state_dict(weights, strict=False)

trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

e = trainer.run(train_loader, max_epochs=epochs)

tb_logger.close()
