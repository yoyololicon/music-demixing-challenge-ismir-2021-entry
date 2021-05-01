import random
import torch
from torch.utils.data import DataLoader
from torch import optim
import argparse
import os
from datetime import datetime
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan, Checkpoint
from ignite.contrib.handlers.tensorboard_logger import *

from dataset import FastMUSDB
from loss import WaveGlowLoss
from model import WaveFlow


parser = argparse.ArgumentParser()

# Dataset paramaters
parser.add_argument('root', type=str, help='root path of dataset')
parser.add_argument('--save', type=str, default="waveflow_saved/")
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--log-dir', type=str, default='waveflow_logs/')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=1)
parser.add_argument('--samples-per-track', type=int, default=16)
parser.add_argument('--cum-steps', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.0002,
                    help='learning rate, defaults to 1e-3')
parser.add_argument('--patience', type=int, default=250,
                    help='minimum number of bad epochs for EarlyStoping (default: 250)')
parser.add_argument('--lr-decay-patience', type=int, default=20,
                    help='lr decay patience for plateau scheduler')
parser.add_argument('--lr-decay-gamma', type=float, default=0.3,
                    help='gamma of learning rate scheduler decay')
parser.add_argument('--weight-decay', type=float, default=0.00001,
                    help='weight decay')
parser.add_argument('--seed', type=int, default=2434, metavar='S',
                    help='random seed (default: 2434)')

# Model Parameters
parser.add_argument('--seq-dur', type=float, default=4.2,
                    help='Sequence duration in seconds per trainig batch'
                    'value of <=0.0 will use full/variable length')
parser.add_argument('--hidden-size', type=int, default=48,
                    help='hidden size parameter of dense bottleneck layers')


args = parser.parse_args()

epochs = args.epochs
batch = args.batch_size
samples_per_track = args.samples_per_track
accumulation_steps = args.cum_steps


if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# train_data = MUSDataset(args.root, subsets='train', split='train',
#                        seq_duration=args.seq_dur, size=epoch_steps * batch)
train_data = FastMUSDB(
    args.root, seq_duration=args.seq_dur, samples_per_track=samples_per_track,
    random=True, random_track_mix=True)
val_data = FastMUSDB(
    args.root, subsets='train', split='valid', seq_duration=args.seq_dur, random=False)

train_loader = DataLoader(train_data, batch, num_workers=1, shuffle=True, drop_last=True,
                          pin_memory=True if device == 'cuda' else False, prefetch_factor=2)


model = WaveFlow(4, 7, 128,
                 dilation_channels=args.hidden_size,
                 residual_channels=args.hidden_size,
                 skip_channels=args.hidden_size).to(device)
optimizer = optim.Adam(model.parameters(), args.lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, verbose=True)
criterion = WaveGlowLoss()


print('Trainable parameters: {}'.format(sum(p.numel()
                                            for p in model.parameters() if p.requires_grad)))


def process_function(engine, batch):
    model.train()

    x, y = batch
    x, y = x.to(device), y.to(device)

    spec = model.get_spec(x)
    z, logdet = model(y, spec)

    loss = criterion(z, logdet)
    loss /= accumulation_steps
    loss.backward()

    if engine.state.iteration % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * accumulation_steps


trainer = Engine(process_function)


RunningAverage(output_transform=lambda x: x).attach(trainer, 'loss')


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    print("Trainer Results - Epoch {} - Avg loss: {:.2f}"
          .format(engine.state.epoch, avg_loss))


# Tqdm
pbar = ProgressBar(persist=True)
pbar.attach(trainer, 'all')

# Create a logger
start_time = datetime.now().strftime('%m%d_%H%M%S')
tb_logger = TensorboardLogger(log_dir=os.path.join(
    args.log_dir, "WaveFlow", start_time))
tb_logger.attach_output_handler(
    trainer,
    event_name=Events.ITERATION_COMPLETED,
    tag="training",
    metric_names='all'
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


checkpointer = ModelCheckpoint(
    args.save, 'waveflow', n_saved=2, create_dir=True, require_empty=False)
to_save = {
    'model': model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'trainer': trainer,
}
trainer.add_event_handler(
    Events.EPOCH_COMPLETED(every=4),
    checkpointer,
    to_save
)


def predict_samples(engine):
    model.eval()
    with torch.no_grad():
        x, _ = val_data[random.randrange(len(val_data))]
        x = torch.from_numpy(x).to(device).unsqueeze(0)

        spec = model.get_spec(x)

        xpred = model.infer(spec, 0.7)[0].squeeze(
        ).transpose(1, 2).cpu().clip(-1, 1)

        tb_logger.writer.add_audio(
            'mixture', x.squeeze().t().cpu(), engine.state.epoch)
        for i in range(4):
            tb_logger.writer.add_audio(
                val_data.sources[i], xpred[i], engine.state.epoch)


trainer.add_event_handler(Events.EPOCH_COMPLETED, predict_samples)

if args.checkpoint:
    checkpoint = torch.load(args.checkpoint)
    Checkpoint.load_objects(to_load=to_save, checkpoint=checkpoint)


trainer.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())

e = trainer.run(train_loader, max_epochs=epochs)

tb_logger.close()
