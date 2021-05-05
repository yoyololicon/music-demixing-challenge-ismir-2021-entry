from collections import namedtuple
import random
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda import amp
import argparse
import json
from datetime import datetime
import os
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, Average
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan, Checkpoint
from ignite.contrib.handlers.tensorboard_logger import *

import dataset as module_data
import loss as module_loss
import model as module_arch

from utils import get_instance


parser = argparse.ArgumentParser(description='SS Trainer')

parser.add_argument('config', type=str, help='config file')
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()

config_file = args.config
checkpoint_file = args.checkpoint
config = json.load(open(config_file))


if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

train_data = get_instance(module_data, config['dataset']['train'])
val_data = get_instance(module_data, config['dataset']['valid'])

train_loader = DataLoader(train_data, **config['data_loader']['train'])
val_loader = DataLoader(val_data, **config['data_loader']['valid'])

model = get_instance(module_arch, config['arch']).to(device)
optimizer = get_instance(optim, config['optimizer'], model.parameters())
scheduler = get_instance(optim.lr_scheduler, config['lr_scheduler'], optimizer)
criterion = get_instance(module_loss, config['loss']).to(device)


print('Trainable parameters: {}'.format(sum(p.numel()
                                            for p in model.parameters() if p.requires_grad)))

trainer_args = namedtuple("trainer_args", config['trainer'].keys())(
    *config['trainer'].values())

# get target index
targets_idx = []
for t in trainer_args.targets:
    targets_idx.append(train_data.sources.index(t))
assert len(targets_idx) > 0
targets_idx = sorted(targets_idx)

amp_enabled = trainer_args.amp_enabled
n_fft = trainer_args.n_fft
hop_length = trainer_args.hop_length
accumulation_steps = trainer_args.cum_steps

scaler = amp.GradScaler(enabled=amp_enabled)
spec = module_arch.Spec(n_fft, hop_length).to(device)


def process_function(engine, batch):
    model.train()

    x, y = batch
    y = y[:, targets_idx].squeeze(1)
    x, y = x.to(device), y.to(device)

    X = spec(x)
    Y = spec(y)
    input = X.abs()
    with amp.autocast(enabled=amp_enabled):
        pred_mask = model(input)

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
        input = X.abs()
        with amp.autocast(enabled=amp_enabled):
            pred_mask = model(input)

        loss, extra_losses = criterion(pred_mask, Y, X, y, x)

        result = {'loss': loss.item()}
        result.update(extra_losses)
        return result


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)

extra_monitor = trainer_args.extra_monitor
RunningAverage(output_transform=lambda x: x['loss']).attach(trainer, 'loss')
Average(output_transform=lambda x: x['loss']).attach(evaluator, 'loss')
for k in extra_monitor:
    RunningAverage(output_transform=lambda x, m=k: x[m]).attach(trainer, k)
    Average(output_transform=lambda x, m=k: x[m]).attach(evaluator, k)


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    output_str = "Trainer Results - Epoch {} - Avg loss: {:.2f}".format(
        engine.state.epoch, avg_loss)
    for k in extra_monitor:
        output_str += " Avg " + k + ": {:.2f}".format(engine.state.metrics[k])
    print(output_str)


def print_logs(engine, dataloader):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_loss = metrics['loss']
    scheduler.step(avg_loss)

    output_str = "Evaluater Results - Epoch {} - Avg loss: {:.2f}".format(
        engine.state.epoch, avg_loss)
    for k in extra_monitor:
        output_str += " Avg " + k + ": {:.2f}".format(metrics[k])
    print(output_str)


validate_every = trainer_args.validate_every
val_epoch_length = trainer_args.val_epoch_length
trainer.add_event_handler(Events.EPOCH_COMPLETED(
    every=validate_every), print_logs, val_loader)

# Tqdm

pbar = ProgressBar(persist=True)
pbar.attach(trainer, 'all')

# Create a logger
model_name = config['name']
log_dir = trainer_args.log_dir
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


# early stop

def score_function(engine):
    return -engine.state.metrics['loss']


patience = trainer_args.patience
handler = EarlyStopping(
    patience=patience, score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)

save_dir = trainer_args.save_dir
checkpointer = ModelCheckpoint(
    save_dir, model_name, n_saved=2, create_dir=True, require_empty=False)
to_save = {
    'model': model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'trainer': trainer,
    'scaler': scaler
}
trainer.add_event_handler(
    Events.EPOCH_COMPLETED,
    checkpointer,
    to_save
)


def predict_samples(engine):
    model.eval()
    with torch.no_grad():
        x, _ = val_data[random.randrange(len(val_data))]
        x = torch.from_numpy(x)
        tb_logger.writer.add_audio('mixture', x.t(), engine.state.epoch)

        X = spec(x.to(device))
        input = X.abs()
        with amp.autocast(enabled=amp_enabled):
            pred_mask = model(input.unsqueeze(0)).squeeze()

        xpred = spec(pred_mask * (X.unsqueeze(1) if len(targets_idx)
                                  > 1 else X), inverse=True).cpu().clip(-1, 1)

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


trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

epochs = trainer_args.epochs

e = trainer.run(train_loader, max_epochs=epochs)

tb_logger.close()
