import torch
import torch.nn.functional as F
from model import Spec
from itertools import combinations, chain
from utils import MWF


class _TLoss(torch.nn.Module):
    r"""Base class for time domain loss modules.

    You can't use this module directly.
    Your loss should also subclass this class.

    Args:
        pred (Tensor): predict time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        gt (Tensor): target time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        mix (Tensor): mixture time signal tensor with size (batch, channels, samples)

    Returns:
        tuple: a length-2 tuple with the first element is the final loss tensor, 
            and the second is a dict containing any intermediate loss value you want to monitor
    """

    def forward(self, *args, **kwargs):
        return self._core_loss(*args, **kwargs)

    def _core_loss(self, pred, gt, mix):
        raise NotImplementedError


class _FLoss(torch.nn.Module):
    r"""Base class for frequency domain loss modules.

    You can't use this module directly.
    Your loss should also subclass this class.

    Args:
        msk_hat (Tensor): mask tensor with size (batch, *, channels, bins, frames), `*` is an optional multi targets dimension.
                        The tensor value should always be non-negative
        gt_spec (Tensor): target spectrogram complex tensor with size (batch, *, channels, bins, frames), `*` is an optional multi targets dimension.
        mix_spec (Tensor): mixture spectrogram complex tensor with size (batch, channels, bins, frames)
        gt (Tensor): target time signal tensor with size (batch, *, channels, samples), `*` is an optional multi targets dimension.
        mix (Tensor): mixture time signal tensor with size (batch, channels, samples)

    Returns:
        tuple: a length-2 tuple with the first element is the final loss tensor, 
            and the second is a dict containing any intermediate loss value you want to monitor
    """

    def forward(self, *args, **kwargs):
        return self._core_loss(*args, **kwargs)

    def _core_loss(self, msk_hat, gt_spec, mix_spec, gt, mix):
        raise NotImplementedError


class SDR(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.expr = 'bi,bi->b'

    def _batch_dot(self, x, y):
        return torch.einsum(self.expr, x, y)

    def forward(self, estimates, references):
        length = min(references.shape[-1], estimates.shape[-1])
        references = references[..., :length].reshape(references.shape[0], -1)
        estimates = estimates[..., :length].reshape(estimates.shape[0], -1)

        delta = 1e-7  # avoid numerical errors
        num = self._batch_dot(references, references)
        den = num + self._batch_dot(estimates, estimates) - \
            2 * self._batch_dot(estimates, references)
        den = den.relu().add_(delta).log10()
        num = num.add_(delta).log10()
        return 10 * (num - den)


class WaveGlowLoss(torch.nn.Module):
    def __init__(self, sigma=1.):
        super().__init__()
        self.sigma2 = sigma ** 2

    def forward(self, z, logdet):
        z = z.reshape(-1)
        loss = 0.5 * z @ z / self.sigma2 - logdet.sum()
        loss = loss / z.numel()
        return loss, {}


class CLoss(_FLoss):
    def __init__(self, mcoeff=10, n_fft=4096, hop_length=1024, complex_mse=True, **mwf_kwargs):
        super().__init__()
        self.mcoeff = mcoeff
        self.spec = Spec(n_fft, hop_length)
        self.complex_mse = complex_mse
        if len(mwf_kwargs):
            self.mwf = MWF(**mwf_kwargs)

    def _core_loss(self, msk_hat, gt_spec, mix_spec, gt, mix):
        if hasattr(self, 'mwf'):
            Y = self.mwf(msk_hat, mix_spec)
        else:
            Y = msk_hat * mix_spec.unsqueeze(1)
        pred = self.spec(Y, inverse=True)
        if self.complex_mse:
            loss_f = complex_mse_loss(Y, gt_spec)
        else:
            loss_f = real_mse_loss(msk_hat, gt_spec.abs(), mix_spec.abs())
        loss_t = sdr_loss(pred, gt, mix)
        loss = loss_f + self.mcoeff * loss_t
        return loss, {
            "loss_f": loss_f.item(),
            "loss_t": loss_t.item()
        }


class MDLoss(_FLoss):
    def __init__(self, mcoeff=10, n_fft=4096, hop_length=1024):
        super().__init__()
        self.mcoeff = mcoeff
        self.spec = Spec(n_fft, hop_length)

    def _core_loss(self, msk_hat, gt_spec, mix_spec, gt, mix):
        pred_spec = msk_hat * mix_spec
        diff = pred_spec - gt_spec

        real = diff.real.reshape(-1)
        imag = diff.imag.reshape(-1)
        mse = real @ real + imag @ imag
        loss_f = mse / real.numel()

        pred = self.spec(pred_spec, inverse=True)
        batch_size, n_channels, length = pred.shape

        # Fix Length
        mix = mix[..., :length].reshape(-1, length)
        gt = gt[..., :length].reshape(-1, length)
        pred = pred.view(-1, length)

        loss_t = _sdr_loss_core(pred, gt, mix) + 1.0
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


def complex_mse_loss(y_hat: torch.Tensor, gt_spec: torch.Tensor):
    assert y_hat.shape == gt_spec.shape
    assert gt_spec.is_complex() and y_hat.is_complex()

    loss = []
    for c in chain(combinations(range(4), 1), combinations(range(4), 2), combinations(range(4), 3)):
        m = sum([y_hat[:, i] for i in c])
        gt = sum([gt_spec[:, i] for i in c])
        diff = m - gt
        real = diff.real.reshape(-1)
        imag = diff.imag.reshape(-1)
        mse = real @ real + imag @ imag
        loss.append(mse / real.numel())

    # All 14 Combination Losses (4C1 + 4C2 + 4C3)
    loss_mse = sum(loss) / len(loss)
    return loss_mse


def real_mse_loss(msk_hat: torch.Tensor, gt_spec: torch.Tensor, mix_spec: torch.Tensor):
    assert msk_hat.shape == gt_spec.shape
    assert msk_hat.is_floating_point() and gt_spec.is_floating_point(
    ) and mix_spec.is_floating_point()
    assert not gt_spec.is_complex() and not mix_spec.is_complex()

    loss = []
    for c in chain(combinations(range(4), 1), combinations(range(4), 2), combinations(range(4), 3)):
        m = sum([msk_hat[:, i] for i in c])
        gt = sum([gt_spec[:, i] for i in c])
        loss.append(F.mse_loss(m * mix_spec, gt))

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
    print(real_mse_loss(mix, pred, gt))
