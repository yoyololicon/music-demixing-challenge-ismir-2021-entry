import torch
import torch.nn.functional as F
from model import Spec
from itertools import combinations, chain


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.):
        super().__init__()
        self.sigma2 = sigma ** 2

    def forward(self, z, logdet):
        z = z.reshape(-1)
        loss = 0.5 * z @ z / self.sigma2 - logdet.sum()
        loss = loss / z.numel()
        return loss, {}


class CLoss(torch.nn.Module):
    def __init__(self, mcoeff=10, n_fft=4096, hop_length=1024):
        super().__init__()
        self.mcoeff = mcoeff
        self.spec = Spec(n_fft, hop_length)

    def forward(self, msk_hat, gt_spec, mix_spec, gt, mix):
        pred = self.spec(msk_hat * mix_spec.unsqueeze(1), inverse=True)
        loss_f = mse_loss(msk_hat, gt_spec, mix_spec)
        loss_t = sdr_loss(pred, gt, mix)
        loss = loss_f + self.mcoeff * loss_t
        return loss, {
            "loss_f": loss_f.item(),
            "loss_t": loss_t.item()
        }


def bce_loss(msk_hat, gt_spec):
    assert msk_hat.shape == gt_spec.shape
    loss = []
    gt_spec_power = gt_spec.abs()
    gt_spec_power *= gt_spec_power
    divider = gt_spec_power.sum(1) + 1e-10
    for c in chain(combinations(range(4), 1), combinations(range(4), 2), combinations(range(4), 3)):
        m = sum([msk_hat[:, i] for i in c])
        gt = sum([gt_spec_power[:, i] for i in c]) / divider
        loss.append(F.binary_cross_entropy(m, gt))

    # All 14 Combination Losses (4C1 + 4C2 + 4C3)
    loss_mse = sum(loss) / len(loss)
    return loss_mse


def mse_loss(msk_hat, gt_spec, mix_spec):

    assert msk_hat.shape == gt_spec.shape
    #mix_spec_mag = mix_spec.abs()
    #gt_spec_mag = gt_spec.abs()
    loss = []
    for c in chain(combinations(range(4), 1), combinations(range(4), 2), combinations(range(4), 3)):
        m = sum([msk_hat[:, i] for i in c])
        gt = sum([gt_spec[:, i] for i in c])
        diff = m * mix_spec - gt
        real = diff.real.reshape(-1)
        imag = diff.imag.reshape(-1)
        mse = real @ real + imag @ imag
        loss.append(mse / real.numel())
        #loss.append(F.mse_loss(m * mix_spec_mag, gt))

    # All 14 Combination Losses (4C1 + 4C2 + 4C3)
    loss_mse = sum(loss) / len(loss)
    return loss_mse


def sdr_loss(pred, gt_time, mix):
    # SDR-Combination Loss

    batch_size, _, n_channels, length = pred.shape
    pred, gt_time = pred.transpose(
        0, 1).contiguous(), gt_time.transpose(0, 1).contiguous()

    # Fix Length
    mix = mix[..., :length].reshape(-1, length)
    gt_time = gt_time[..., :length].reshape(_, -1, length)
    pred = pred.view(_, -1, length)

    extend_pred = [pred.view(-1, length)]
    extend_gt = [gt_time.view(-1, length)]

    for c in chain(combinations(range(4), 2), combinations(range(4), 3)):
        extend_pred.append(sum([pred[i] for i in c]))
        extend_gt.append(sum([gt_time[i] for i in c]))

    extend_pred = torch.cat(extend_pred, 0)
    extend_gt = torch.cat(extend_gt, 0)
    extend_mix = mix.repeat(14, 1)

    loss_sdr = _sdr_loss_core(extend_pred, extend_gt, extend_mix)

    return 1.0 + loss_sdr


def _sdr_loss_core(x_hat, x, y):
    assert x.shape == y.shape == x_hat.shape  # (Batch, Len)

    ns = y - x
    ns_hat = y - x_hat

    ns_norm = ns[:, None, :] @ ns[:, :, None]
    ns_hat_norm = ns_hat[:, None, :] @ ns_hat[:, :, None]

    x_norm = x[:, None, :] @ x[:, :, None]
    x_hat_norm = x_hat[:, None, :] @ x_hat[:, :, None]
    x_cross = x[:, None, :] @ x_hat[:, :, None]

    x_norm, x_hat_norm, ns_norm, ns_hat_norm = x_norm.relu(
    ), x_hat_norm.relu(), ns_norm.relu(), ns_hat_norm.relu()

    alpha = x_norm / (ns_norm + x_norm + 1e-10)

    # Target
    sdr_cln = x_cross / (x_norm.sqrt() * x_hat_norm.sqrt() + 1e-10)

    # Noise
    num_noise = ns[:, None, :] @ ns_hat[:, :, None]
    denom_noise = ns_norm.sqrt() * ns_hat_norm.sqrt()
    sdr_noise = num_noise / (denom_noise + 1e-10)

    return torch.mean(-alpha * sdr_cln - (1 - alpha) * sdr_noise)


if __name__ == "__main__":
    mix, pred, gt_time = torch.randn(2, 2, 100), torch.randn(
        4, 2, 2, 100, requires_grad=True), torch.randn(4, 2, 2, 100)
    print(sdr_loss(mix, pred, gt_time))

    mix, pred, gt = torch.rand(2, 2, 2049, 100), torch.rand(
        4, 2, 2, 2049, 100, requires_grad=True), torch.rand(4, 2, 2, 2049, 100)
    print(mse_loss(mix, pred, gt))
