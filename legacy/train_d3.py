import random
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from torch.cuda import amp
import argparse
import json
from datetime import datetime
import os
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage, MeanSquaredError, Loss
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan, Checkpoint
from ignite.contrib.handlers.tensorboard_logger import *
from torch_optimizer import RAdam, Yogi

from dataset import FastMUSDB
from model import get_vocals_model, Spec

parser = argparse.ArgumentParser(description='D3Net Trainer')

parser.add_argument('config', type=str, help='config file')
parser.add_argument('--checkpoint', type=str, default=None)

args = parser.parse_args()

config_file = args.config
checkpoint_file = args.checkpoint
config = json.load(open(config_file))

model_name = config['name']
epochs = config['epochs']
root = config['root']
root = os.path.expanduser(root)
log_dir = config['log_dir']
save_dir = config['save']
batch = config['batch_size']
samples_per_track = config['samples_per_track']
accumulation_steps = config['cum_steps']
seq_dur = config['seq_dur']


if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

train_data = FastMUSDB(
    root, subsets='train', split='train', seq_duration=seq_dur, samples_per_track=samples_per_track,
    random=True, random_track_mix=True)
val_data = FastMUSDB(
    root, subsets='train', split='valid', seq_duration=seq_dur, random=False)

train_loader = DataLoader(train_data, batch, num_workers=1, shuffle=True, drop_last=True,
                          pin_memory=True if device == 'cuda' else False, prefetch_factor=2)
val_loader = DataLoader(val_data, batch, num_workers=1,
                        pin_memory=True if device == 'cuda' else False, prefetch_factor=2)

sr = train_data.sr
weight_decay = config['weight_decay']
lr = config['lr']
lr_decay_gamma = config['lr_decay_gamma']
lr_decay_patience = config['lr_decay_patience']
patience = config['patience']
amp_enabled = config['amp_enabled']
target_name = config['target']

model = get_vocals_model().to(device)
optimizer = optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=lr_decay_gamma, patience=lr_decay_patience, verbose=True)
scaler = amp.GradScaler(enabled=amp_enabled)
spec = Spec(4096, 1024).to(device)

target_idx = 0
for i, s in enumerate(train_data.sources):
    if s == target_name:
        target_idx = i
        break

print('Trainable parameters: {}'.format(sum(p.numel()
                                            for p in model.parameters() if p.requires_grad)))


def process_function(engine, batch):
    model.train()

    x, y = batch
    y = y[:, target_idx]
    x, y = x.to(device), y.to(device)

    X = spec(x).abs()
    Y = spec(y).abs()
    logX = X.add(1e-12).log()
    with amp.autocast(enabled=amp_enabled):
        mask_logits = model(logX)
    mask_logits = mask_logits.float()
    loss = F.mse_loss(mask_logits.sigmoid() * X, Y)

    loss /= accumulation_steps
    scaler.scale(loss).backward()

    if engine.state.iteration % accumulation_steps == 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return loss.item() * accumulation_steps


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        y = y[:, target_idx]
        x, y = x.to(device), y.to(device)

        X = spec(x).abs()
        Y = spec(y).abs()
        logX = X.add(1e-12).log()
        with amp.autocast(enabled=amp_enabled):
            mask_logits = model(logX)
        mask_logits = mask_logits.float()

        return mask_logits.sigmoid() * X, Y


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)

RunningAverage(output_transform=lambda x: x,
               device=device).attach(trainer, 'loss')
#MeanSquaredError(device=device).attach(evaluator, "loss")
Loss(F.mse_loss, device=device).attach(evaluator, "loss")


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    print("Trainer Results - Epoch {} - Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_loss))


def print_logs(engine, dataloader):
    evaluator.run(dataloader, max_epochs=1, epoch_length=100)
    metrics = evaluator.state.metrics
    avg_loss = metrics['loss']

    scheduler.step(avg_loss)

    print("Evaluater Results - Epoch {} - Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_loss))


trainer.add_event_handler(Events.EPOCH_COMPLETED(
    every=1), print_logs, val_loader)

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

tb_logger.writer.add_graph(
    model, input_to_model=torch.rand(1, 2, 2049, 256, device=device))

# early stop


def score_function(engine):
    return -engine.state.metrics['loss']


handler = EarlyStopping(patience=patience,
                        score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)


checkpointer = ModelCheckpoint(
    save_dir, model_name, n_saved=2, create_dir=True, require_empty=False)
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
        x = torch.from_numpy(x).to(device).unsqueeze(0)

        X = spec(x)
        logX = X.abs().add(1e-12).log()
        with amp.autocast(enabled=amp_enabled):
            mask_logits = model(logX)
        mask_logits = mask_logits.float()
        xpred = spec(mask_logits.sigmoid() * X,
                     inverse=True).squeeze().t().cpu().clip(-1, 1)

        tb_logger.writer.add_audio(
            'mixture', x.squeeze().t().cpu(), engine.state.epoch)

        tb_logger.writer.add_audio(
            val_data.sources[target_idx], xpred, engine.state.epoch)


trainer.add_event_handler(Events.EPOCH_COMPLETED, predict_samples)

if checkpoint_file:
    checkpoint = torch.load(checkpoint_file)
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)


trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

e = trainer.run(train_loader, max_epochs=epochs)

tb_logger.close()
