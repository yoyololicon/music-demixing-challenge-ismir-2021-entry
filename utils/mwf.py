import torch
import norbert


class MWF(torch.nn.Module):
    def __init__(self,
                 residual_model=False,
                 softmask=False,
                 alpha=1.0,
                 n_iter=1) -> None:
        super().__init__()
        self.residual_model = residual_model
        self.n_iter = n_iter
        self.softmask = softmask
        self.alpha = alpha

    def forward(self, msk_hat, mix_spec):
        V = msk_hat * mix_spec.abs().unsqueeze(1)
        if self.softmask and self.alpha != 1:
            V = V.pow(self.alpha)

        X = mix_spec.transpose(1, 3).contiguous()
        V = V.permute(0, 4, 3, 2, 1).contiguous()

        if self.residual_model or V.shape[1] == 1:
            V = norbert.residual_model(
                V, X, self.alpha if self.softmask else 1)

        Y = norbert.wiener(V, X.to(torch.complex128),
                           self.n_iter, use_softmask=self.softmask)
        
        Y = Y.permute(0, 4, 3, 2, 1).contiguous()
        return Y
