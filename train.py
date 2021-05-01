import random
import torch
from torch.utils.data import DataLoader
from torch import optim
import argparse
from ignite.engine import Engine, Events
from ignite.metrics import Loss, RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.handlers import EarlyStopping, ModelCheckpoint, TerminateOnNan, Checkpoint
from ignite.contrib.handlers.tensorboard_logger import *

from dataset import FastMUSDB
from loss import bce_loss, mse_loss, sdr_loss
from model import X_UMX

parser = argparse.ArgumentParser(
    description='OpenUnmix_CrossNet(X-UMX) Trainer')

# Dataset paramaters
parser.add_argument('root', type=str, help='root path of dataset')
parser.add_argument('--save', type=str, default="saved/")
parser.add_argument('--checkpoint', type=str, default=None)
parser.add_argument('--log-dir', type=str, default='logs/')

# Trainig Parameters
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=8)
parser.add_argument('--samples-per-track', type=int, default=64)
parser.add_argument('--cum-steps', type=int, default=2)
parser.add_argument('--lr', type=float, default=0.001,
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
parser.add_argument('--seq-dur', type=float, default=6.0,
                    help='Sequence duration in seconds per trainig batch'
                    'value of <=0.0 will use full/variable length')
parser.add_argument('--nfft', type=int, default=4096,
                    help='STFT fft size and window size')
parser.add_argument('--nhop', type=int, default=1024,
                    help='STFT hop size')
parser.add_argument('--hidden-size', type=int, default=512,
                    help='hidden size parameter of dense bottleneck layers')
parser.add_argument('--bandwidth', type=int, default=16000,
                    help='maximum model bandwidth in herz')

# Misc Parameters
parser.add_argument('--mcoef', type=float, default=10.0,
                    help='coefficient for mixing: mfoef*TD-Loss + FD-Loss')

# The duration of validation sample
parser.add_argument('--valid-dur', type=float, default=10.0,
                    help='Split duration for validation sample to avoid GPU memory overflow')

args = parser.parse_args()

epochs = args.epochs
batch = args.batch_size
samples_per_track = args.samples_per_track
accumulation_steps = args.cum_steps
mcoef = args.mcoef


if torch.cuda.is_available():
    device = 'cuda'
    torch.backends.cudnn.benchmark = True
else:
    device = 'cpu'

# train_data = MUSDataset(args.root, subsets='train', split='train',
#                        seq_duration=args.seq_dur, size=epoch_steps * batch)
train_data = FastMUSDB(
    args.root, subsets='train', split='train', seq_duration=args.seq_dur, samples_per_track=samples_per_track,
    random=True, random_track_mix=True)
val_data = FastMUSDB(
    args.root, subsets='train', split='valid', seq_duration=args.seq_dur, random=False)

train_loader = DataLoader(train_data, batch, num_workers=2, shuffle=True, drop_last=True,
                          pin_memory=True if device == 'cuda' else False, prefetch_factor=4)
val_loader = DataLoader(val_data, batch, num_workers=2,
                        pin_memory=True if device == 'cuda' else False, prefetch_factor=4)

sr = train_data.sr
max_bins = int(args.bandwidth / sr * args.nfft) + 1

model = X_UMX(args.nfft, args.nhop, args.hidden_size,
              max_bins, 2, 3).to(device)
optimizer = optim.Adam(model.parameters(), args.lr,
                       weight_decay=args.weight_decay)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=args.lr_decay_gamma, patience=args.lr_decay_patience, verbose=True)


print('Trainable parameters: {}'.format(sum(p.numel()
                                            for p in model.parameters() if p.requires_grad)))


def combine_loss(loss_f, loss_t):
    return loss_f + mcoef * loss_t


def process_function(engine, batch):
    model.train()

    x, y = batch
    x, y = x.to(device), y.to(device)

    X = model.t2f(x)
    Y = model.t2f(y)
    Xmag = X.abs()
    pred_mask = model(Xmag)

    xpred = model.f2t(pred_mask * X.unsqueeze(1), length=x.shape[-1])
    loss_f = mse_loss(pred_mask, Y, X)
    loss_t = sdr_loss(xpred, y, x)
    loss = combine_loss(loss_f, loss_t)
    loss /= accumulation_steps
    loss.backward()

    if engine.state.iteration % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    return loss.item() * accumulation_steps, loss_f.item(), loss_t.item()


def evaluate_function(engine, batch):
    model.eval()
    with torch.no_grad():
        x, y = batch
        x, y = x.to(device), y.to(device)

        X = model.t2f(x)
        Y = model.t2f(y)
        Xmag = X.abs()
        pred_mask = model(Xmag)

        xpred = model.f2t(pred_mask * X.unsqueeze(1), length=x.shape[-1])
        return xpred, y, x, pred_mask, Y, X


trainer = Engine(process_function)
evaluator = Engine(evaluate_function)

RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'loss')
RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'loss_f')
RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss_t')
#RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'loss_f(bce)')

Loss(mse_loss, output_transform=lambda x: [
     x[3], x[4], {"mix_spec": x[5]}]).attach(evaluator, 'loss_f')
Loss(sdr_loss, output_transform=lambda x: [
     x[0], x[1], {"mix": x[2]}]).attach(evaluator, 'loss_t')
# Loss(bce_loss, output_transform=lambda x: [
#     x[3], x[4]]).attach(evaluator, 'loss_f(bce)')


@trainer.on(Events.EPOCH_COMPLETED)
def print_trainer_logs(engine):
    avg_loss = engine.state.metrics['loss']
    avg_loss_f = engine.state.metrics['loss_f']
    avg_loss_t = engine.state.metrics['loss_t']
    print("Trainer Results - Epoch {} - Avg loss: {:.2f} Avg loss_f: {:.2f} Avg loss_t: {:.2f}"
          .format(engine.state.epoch, avg_loss, avg_loss_f, avg_loss_t))


def print_logs(engine, dataloader):
    evaluator.run(dataloader, max_epochs=1)
    metrics = evaluator.state.metrics
    avg_loss_f = metrics['loss_f']
    avg_loss_t = metrics['loss_t']
    avg_loss = combine_loss(avg_loss_f, avg_loss_t)

    scheduler.step(avg_loss)

    print("Evaluater Results - Epoch {} - Avg loss: {:.2f} Avg loss_f: {:.2f} Avg loss_t: {:.2f}"
          .format(engine.state.epoch, avg_loss, avg_loss_f, avg_loss_t))


trainer.add_event_handler(Events.EPOCH_COMPLETED(
    every=4), print_logs, val_loader)

# Tqdm

pbar = ProgressBar(persist=True)
pbar.attach(trainer, 'all')

# Create a logger
tb_logger = TensorboardLogger()
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
    return -combine_loss(engine.state.metrics['loss_f'], engine.state.metrics['loss_t'])


handler = EarlyStopping(patience=args.patience,
                        score_function=score_function, trainer=trainer)
evaluator.add_event_handler(Events.COMPLETED, handler)


checkpointer = ModelCheckpoint(
    args.save, 'x-umx', n_saved=2, create_dir=True, require_empty=False,
    score_function=score_function, score_name="negative_loss")
to_save = {
    'model': model,
    'optimizer': optimizer,
    'scheduler': scheduler,
    'trainer': trainer,
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

        X = model.t2f(x)
        Xmag = X.abs()
        pred_mask = model(Xmag)

        xpred = model.f2t(pred_mask * X.unsqueeze(1),
                          length=x.shape[-1]).squeeze().transpose(1, 2).cpu().clip(-1, 1)
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
