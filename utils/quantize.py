import torch
import torch.nn.functional as F
from torch.autograd import Function


# Brought from https://github.com/distsup/DistSup/blob/b05ef514cbb8863e28cd3c41b267c5b9d6226a82/distsup/modules/bottlenecks.py#L602
class VectorQuantization(Function):

    @staticmethod
    def flatten(x):
        code_dim = x.size(-1)
        return x.view(-1, code_dim)

    @staticmethod
    def restore_shapes(codes, indices, target_shape):
        idx_shape = list(target_shape)
        idx_shape[-1] = 1
        return codes.view(*target_shape), indices.view(*idx_shape)

    @staticmethod
    def forward(ctx, inputs, codebook, commitment=0.25, use_copy_through=False):
        inputs_flat = VectorQuantization.flatten(inputs)

        # TODO: get the indices according to distances
        indices = get_indices(inputs_flat, codebook)
        codes = codebook[indices.view(-1), :]
        codes, indices = VectorQuantization.restore_shapes(
            codes, indices, inputs.shape
        )

        ctx.save_for_backward(
            codes, inputs, torch.FloatTensor([commitment]),
            codebook, indices, torch.tensor([use_copy_through])
        )
        ctx.mark_non_differentiable(indices)
        return codes, indices

    @staticmethod
    def backward(ctx, straight_through, unused_indices):
        (codes, inputs, beta, codebook, indices, use_copy_through
         ) = ctx.saved_tensors

        # gradient of the l2 loss
        # +diff for inputs and -diff for codes
        diff = 2 * (inputs - codes) / inputs.numel()

        # commitment loss for updating inputs
        commitment = beta.item() * diff

        # determine gradient for codebook
        if use_copy_through.item():
            # update codebook with both vq_loss and reconstruction loss
            code_disp = VectorQuantization.flatten(-diff + straight_through)
        else:
            # update codebook with only vq_loss
            code_disp = VectorQuantization.flatten(-diff)
        # TODO: makes sense to only update codebook with straight_through? 
        # https://github.com/rosinality/vq-vae-2-pytorch/blob/ef5f67c46f93624163776caec9e0d95063910eca/vqvae.py#L71

        indices = VectorQuantization.flatten(indices)
        code_disp = (torch
                     .zeros_like(codebook)
                     .index_add_(0, indices.view(-1), code_disp))

        # number of variables is equal to number of input arguments to forward()
        return straight_through + commitment, code_disp, None, None


quantize = VectorQuantization.apply


# Brought from https://github.com/rosinality/vq-vae-2-pytorch/blob/ef5f67c46f93624163776caec9e0d95063910eca/vqvae.py#L43
# def get_indices(inputs_flat, codebook):
#     dist = (
#         inputs_flat.pow(2).sum(1, keepdim=True)
#         - 2 * inputs_flat @ codebook
#         + codebook.pow(2).sum(0, keepdim=True)
#     )
#     _, embed_ind = (-dist).max(1)
# 
#     return embed_ind


# Brought from https://github.com/distsup/DistSup/blob/b05ef514cbb8863e28cd3c41b267c5b9d6226a82/distsup/modules/bottlenecks.py#L575
def get_indices(inputs, codebook, temperature=None):
    with torch.no_grad():
        # inputs: NxD
        # codebook: KxD
        # NxK
        distances_matrix = torch.cdist(inputs, codebook)
        # Nx1
        if temperature is None:
            indices = torch.min(distances_matrix, dim=-1)[1].unsqueeze(1)
        else:
            probs = F.softmax(-distances_matrix / temperature, dim=-1)
            m = torch.distributions.Categorical(probs)
            indices = m.sample()
        return indices
