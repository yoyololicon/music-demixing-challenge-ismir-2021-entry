import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.activation import MultiheadAttention
import copy


class GlobalLayerNorm(nn.Module):
    """Calculate Global Layer Normalization.

    Arguments
    ---------
       dim : (int or list or torch.Size)
           Input shape from an expected input of size.
       eps : float
           A value added to the denominator for numerical stability.
       elementwise_affine : bool
          A boolean value that when set to True,
          this module has learnable per-element affine parameters
          initialized to ones (for weights) and zeros (for biases).

    Example
    -------
    >>> x = torch.randn(5, 10, 20)
    >>> GLN = GlobalLayerNorm(10, 3)
    >>> x_norm = GLN(x)
    """

    def __init__(self, dim, shape, eps=1e-8, elementwise_affine=True):
        super(GlobalLayerNorm, self).__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            if shape == 3:
                self.weight = nn.Parameter(torch.ones(self.dim, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1))
            if shape == 4:
                self.weight = nn.Parameter(torch.ones(self.dim, 1, 1))
                self.bias = nn.Parameter(torch.zeros(self.dim, 1, 1))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    def forward(self, x):
        """Returns the normalized tensor.

        Arguments
        ---------
        x : torch.Tensor
            Tensor of size [N, C, K, S] or [N, C, L].
        """
        # x = N x C x K x S or N x C x L
        # N x 1 x 1
        # cln: mean,var N x 1 x K x S
        # gln: mean,var N x 1 x 1
        if x.dim() == 3:
            mean = torch.mean(x, (1, 2), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)

        if x.dim() == 4:
            mean = torch.mean(x, (1, 2, 3), keepdim=True)
            var = torch.mean((x - mean) ** 2, (1, 2, 3), keepdim=True)
            if self.elementwise_affine:
                x = (
                    self.weight * (x - mean) / torch.sqrt(var + self.eps)
                    + self.bias
                )
            else:
                x = (x - mean) / torch.sqrt(var + self.eps)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1) # seq_len, batch, channels
        pe = pe.transpose(0, 1).unsqueeze(0) # batch, channels, seq_len
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x is seq_len, batch, channels
        # x = x + self.pe[:x.size(0), :]

        # x is batch, channels, seq_len
        x = x + self.pe[:, :, :x.size(2)]
        return self.dropout(x)
    
class TransformerLayer(nn.Module):
    """
    Transformer layer.
    Args:
        embed_dim (int): Number of input channels.
        n_heads (int): Number of attention heads.
        dim_ff (int): Number of neurons in the RNNs cell state.
            Defaults to 256. RNN here replaces standard FF linear layer in plain Transformer.
        dropout (float, optional): Dropout ratio, must be in [0,1].
        activation (str, optional): activation function applied at the output of RNN.
        norm (str, optional): Type of normalization to use.
    References
        [1] Cem Subakan, Mirco Ravanelli, Samuele Cornell, Mirko Bronzi, and 
        Jianyuan Zhong. "Attention is All You Need in Speech Separation."
        arXiv (2020).
    """

    def __init__(
        self,
        embed_dim,
        n_heads,
        dim_ff=2048,
        dropout=0.0,
    ):
        super(TransformerLayer, self).__init__()

        self.mha = MultiheadAttention(embed_dim, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(embed_dim, dim_ff)
        self.linear2 = nn.Linear(dim_ff, embed_dim)
        self.activation = nn.ReLU()
        #self.norm_mha = GlobalLayerNorm(embed_dim, 4)
        #self.norm_ff = GlobalLayerNorm(embed_dim, 4)

    def forward(self, x):
        #assert x.shape[0] != x.shape[1], "seq_len == channels would lead to wrong LN dimension"
        #tomha = self.norm_mha(x)
        tomha = x.permute(1, 0, 2)
        # x is batch, channels, seq_len
        # mha is seq_len, batch, channels
        # self-attention is applied
        out = self.mha(tomha, tomha, tomha)[0]
        x = self.dropout(out.permute(1, 0, 2)) + x
        out = out.permute(1, 0, 2)

        # lstm is applied
        #toff = self.norm_ff(x)
        out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.dropout(out) + x
        return x

class Encoder(nn.Module):
    """Convolutional Encoder Layer.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.

    Example
    -------
    >>> x = torch.randn(2, 1000)
    >>> encoder = Encoder(kernel_size=4, out_channels=64)
    >>> h = encoder(x)
    >>> h.shape
    torch.Size([2, 64, 499])
    """

    def __init__(self, kernel_size=2, out_channels=64, in_channels=1):
        super(Encoder, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=kernel_size // 2,
            groups=1,
            bias=False,
        )
        self.in_channels = in_channels

    def forward(self, x):
        """Return the encoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, L].
        Return
        ------
        x : torch.Tensor
            Encoded tensor with dimensionality [B, N, T_out].

        where B = Batchsize
              L = Number of timepoints
              N = Number of filters
              T_out = Number of timepoints at the output of the encoder
        """
        # B x L -> B x 1 x L
        if self.in_channels == 1:
            x = torch.unsqueeze(x, dim=1)
        # B x 1 x L -> B x N x T_out
        x = self.conv1d(x)
        x = F.relu(x)

        return x

class Decoder(nn.ConvTranspose1d):
    """A decoder layer that consists of ConvTranspose1d.

    Arguments
    ---------
    kernel_size : int
        Length of filters.
    in_channels : int
        Number of  input channels.
    out_channels : int
        Number of output channels.


    Example
    ---------
    >>> x = torch.randn(2, 100, 1000)
    >>> decoder = Decoder(kernel_size=4, in_channels=100, out_channels=1)
    >>> h = decoder(x)
    >>> h.shape
    torch.Size([2, 1003])
    """

    def __init__(self, *args, **kwargs):
        super(Decoder, self).__init__(*args, **kwargs)

    def forward(self, x):
        """Return the decoded output.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor with dimensionality [B, N, L].
                where, B = Batchsize,
                       N = number of filters
                       L = time points
        """

        if x.dim() not in [2, 3]:
            raise RuntimeError(
                "{} accept 3/4D tensor as input".format(self.__name__)
            )
        x = super().forward(x if x.dim() == 3 else torch.unsqueeze(x, 1))

        if torch.squeeze(x).dim() == 1:
            x = torch.squeeze(x, dim=1)
        else:
            x = torch.squeeze(x)
        return x

class Dual_Computation_Block(nn.Module):
    """Computation block for dual-path processing.

    Arguments
    ---------
    intra_mdl : torch.nn.module
        Model to process within the chunks.
     inter_mdl : torch.nn.module
        Model to process across the chunks.
     out_channels : int
        Dimensionality of inter/intra model.
     norm : str
        Normalization type.
     skip_around_intra : bool
        Skip connection around the intra layer.
     linear_layer_after_inter_intra : bool
        Linear layer or not after inter or intra.

    Example
    ---------
        >>> intra_block = SBTransformerBlock(1, 64, 8)
        >>> inter_block = SBTransformerBlock(1, 64, 8)
        >>> dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
        >>> x = torch.randn(10, 64, 100, 10)
        >>> x = dual_comp_block(x)
        >>> x.shape
        torch.Size([10, 64, 100, 10])
    """

    def __init__(
        self,
        intra_mdl,
        inter_mdl,
        out_channels,
        norm="ln",
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
    ):
        super(Dual_Computation_Block, self).__init__()

        self.intra_mdl = intra_mdl
        self.inter_mdl = inter_mdl
        self.skip_around_intra = skip_around_intra
        self.linear_layer_after_inter_intra = linear_layer_after_inter_intra

        # Norm
        self.norm = norm
        if norm is not None:
            self.intra_norm = GlobalLayerNorm(out_channels, 4)
            self.inter_norm = GlobalLayerNorm(out_channels, 4)

        # Linear
        if linear_layer_after_inter_intra:
            self.intra_linear = nn.Linear(
                out_channels, out_channels
            )

            self.inter_linear = nn.Linear(
                out_channels, out_channels
            )

    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, K, S].


        Return
        ---------
        out: torch.Tensor
            Output tensor of dimension [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
        """
        B, N, K, S = x.shape
        # intra RNN
        # [BS, K, N]
        intra = x.permute(0, 3, 2, 1).contiguous().view(B * S, K, N)
        # [BS, K, H]

        intra = self.intra_mdl(intra)

        # [BS, K, N]
        if self.linear_layer_after_inter_intra:
            intra = self.intra_linear(intra)

        # [B, S, K, N]
        intra = intra.view(B, S, K, N)
        # [B, N, K, S]
        intra = intra.permute(0, 3, 2, 1).contiguous()
        if self.norm is not None:
            intra = self.intra_norm(intra)
        # [B, N, K, S]
        if self.skip_around_intra:
            intra = intra + x

        # inter RNN
        # [BK, S, N]
        inter = intra.permute(0, 2, 3, 1).contiguous().view(B * K, S, N)
        # [BK, S, H]
        inter = self.inter_mdl(inter)

        # [BK, S, N]
        if self.linear_layer_after_inter_intra:
            inter = self.inter_linear(inter)

        # [B, K, S, N]
        inter = inter.view(B, K, S, N)
        # [B, N, K, S]
        inter = inter.permute(0, 3, 1, 2).contiguous()
        if self.norm is not None:
            inter = self.inter_norm(inter)
        # [B, N, K, S]
        out = inter + intra

        return out




class Dual_Path_Model(nn.Module):
    """The dual path model which is the basis for dualpathrnn, sepformer, dptnet.

    Arguments
    ---------
    in_channels : int
        Number of channels at the output of the encoder.
    out_channels : int
        Number of channels that would be inputted to the intra and inter blocks.
    intra_model : torch.nn.module
        Model to process within the chunks.
    inter_model : torch.nn.module
        model to process across the chunks,
    num_layers : int
        Number of layers of Dual Computation Block.
    norm : str
        Normalization type.
    K : int
        Chunk length.
    num_spks : int
        Number of sources (speakers).
    skip_around_intra : bool
        Skip connection around intra.
    linear_layer_after_inter_intra : bool
        Linear layer after inter and intra.
    use_global_pos_enc : bool
        Global positional encodings.
    max_length : int
        Maximum sequence length.

    Example
    ---------
    >>> intra_block = SBTransformerBlock(1, 64, 8)
    >>> inter_block = SBTransformerBlock(1, 64, 8)
    >>> dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=2)
    >>> x = torch.randn(10, 64, 2000)
    >>> x = dual_path_model(x)
    >>> x.shape
    torch.Size([2, 10, 64, 2000])
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_model,
        inter_model,
        num_layers=1,
        norm="ln",
        K=200,
        num_spks=2,
        skip_around_intra=True,
        linear_layer_after_inter_intra=True,
        use_global_pos_enc=False,
        max_length=20000,
    ):
        super(Dual_Path_Model, self).__init__()
        self.K = K
        self.num_spks = num_spks
        self.num_layers = num_layers
        self.norm = GlobalLayerNorm(in_channels, 3)
        self.conv1d = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.use_global_pos_enc = use_global_pos_enc

        if self.use_global_pos_enc:
            self.pos_enc = PositionalEncoding(max_length)

        self.dual_mdl = nn.ModuleList([])
        for i in range(num_layers):
            self.dual_mdl.append(
                copy.deepcopy(
                    Dual_Computation_Block(
                        intra_model,
                        inter_model,
                        out_channels,
                        norm,
                        skip_around_intra=skip_around_intra,
                        linear_layer_after_inter_intra=linear_layer_after_inter_intra,
                    )
                )
            )

        self.conv2d = nn.Conv2d(
            out_channels, out_channels * num_spks, kernel_size=1
        )
        self.end_conv1x1 = nn.Conv1d(out_channels, in_channels, 1, bias=False)
        self.prelu = nn.PReLU()
        self.activation = nn.ReLU()
        # gated output layer
        self.output = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Tanh()
        )
        self.output_gate = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, 1), nn.Sigmoid()
        )


    def forward(self, x):
        """Returns the output tensor.

        Arguments
        ---------
        x : torch.Tensor
            Input tensor of dimension [B, N, L].

        Returns
        -------
        out : torch.Tensor
            Output tensor of dimension [spks, B, N, L]
            where, spks = Number of speakers
               B = Batchsize,
               N = number of filters
               L = the number of time points
        """

        # before each line we indicate the shape after executing the line

        # [B, N, L]
        x = self.norm(x)

        # [B, N, L]
        x = self.conv1d(x)
        if self.use_global_pos_enc:
            x = self.pos_enc(x.transpose(1, -1)).transpose(1, -1) + x * (
                x.size(1) ** 0.5
            )

        # [B, N, K, S]
        x, gap = self._Segmentation(x, self.K)

        # [B, N, K, S]
        for i in range(self.num_layers):
            x = self.dual_mdl[i](x)
        x = self.prelu(x)

        # [B, N*spks, K, S]
        x = self.conv2d(x)
        B, _, K, S = x.shape

        # [B*spks, N, K, S]
        x = x.view(B * self.num_spks, -1, K, S)

        # [B*spks, N, L]
        x = self._over_add(x, gap)
        x = self.output(x) * self.output_gate(x)

        # [B*spks, N, L]
        x = self.end_conv1x1(x)

        # [B, spks, N, L]
        _, N, L = x.shape
        x = x.view(B, self.num_spks, N, L)
        x = self.activation(x)

        # [spks, B, N, L]
        x = x.transpose(0, 1)

        return x

    def _padding(self, input, K):
        """Padding the audio times.

        Arguments
        ---------
        K : int
            Chunks of length.
        P : int
            Hop size.
        input : torch.Tensor
            Tensor of size [B, N, L].
            where, B = Batchsize,
                   N = number of filters
                   L = time points
        """
        B, N, L = input.shape
        P = K // 2
        gap = K - (P + L % K) % K
        if gap > 0:
            pad = torch.Tensor(torch.zeros(B, N, gap)).type(input.type())
            input = torch.cat([input, pad], dim=2)

        _pad = torch.Tensor(torch.zeros(B, N, P)).type(input.type())
        input = torch.cat([_pad, input, _pad], dim=2)

        return input, gap

    def _Segmentation(self, input, K):
        """The segmentation stage splits

        Arguments
        ---------
        K : int
            Length of the chunks.
        input : torch.Tensor
            Tensor with dim [B, N, L].

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, K, S].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points
        """
        B, N, L = input.shape
        P = K // 2
        input, gap = self._padding(input, K)
        # [B, N, K, S]
        input1 = input[:, :, :-P].contiguous().view(B, N, -1, K)
        input2 = input[:, :, P:].contiguous().view(B, N, -1, K)
        input = (
            torch.cat([input1, input2], dim=3).view(B, N, -1, K).transpose(2, 3)
        )

        return input.contiguous(), gap

    def _over_add(self, input, gap):
        """Merge the sequence with the overlap-and-add method.

        Arguments
        ---------
        input : torch.tensor
            Tensor with dim [B, N, K, S].
        gap : int
            Padding length.

        Return
        -------
        output : torch.tensor
            Tensor with dim [B, N, L].
            where, B = Batchsize,
               N = number of filters
               K = time points in each chunk
               S = the number of chunks
               L = the number of time points

        """
        B, N, K, S = input.shape
        P = K // 2
        # [B, N, S, K]
        input = input.transpose(2, 3).contiguous().view(B, N, -1, K * 2)

        input1 = input[:, :, :, :K].contiguous().view(B, N, -1)[:, :, P:]
        input2 = input[:, :, :, K:].contiguous().view(B, N, -1)[:, :, :-P]
        input = input1 + input2
        # [B, N, L]
        if gap > 0:
            input = input[:, :, :-gap]

        return input




if __name__ == "__main__":
    enc = Encoder(kernel_size=16, in_channels=2)
    dec = Decoder(kernel_size=16, in_channels=64, out_channels=2, stride=8)

    spec = torch.rand(8, 2, 256)

    enc_out = enc(spec)
    print(enc_out.shape)
    # torch.Size([8, 64, 255]) -> [B, Channel, L]
    x = dec(enc_out)
    print(x.shape)
    # torch.Size([8, 2, 258]) -> [B, Channel, L]

    """
    x = torch.randn(10, 64, 100, 10)
    intra_block = TransformerLayer(64, 8)
    inter_block = TransformerLayer(64, 8)
    dual_comp_block = Dual_Computation_Block(intra_block, inter_block, 64)
    x = dual_comp_block(x)
    print(x.shape)
    #torch.Size([10, 64, 100, 10])
    """

    intra_block = nn.Sequential(*[TransformerLayer(64, 8) for _ in range(8)]) #TransformerLayer(64, 8)
    inter_block = nn.Sequential(*[TransformerLayer(64, 8) for _ in range(8)]) #TransformerLayer(64, 8)
    dual_path_model = Dual_Path_Model(64, 64, intra_block, inter_block, num_spks=4)
    #x = torch.randn(10, 64, 2000)
    mask = dual_path_model(enc_out)
    print(mask.shape)
    #torch.Size([4, 8, 64, 255])

    mix_w = torch.stack([enc_out] * 4)
    h = sep_h = mix_w * mask
    print(h.shape)
    #torch.Size([4, 8, 64, 255])

    # Decoding
    est_source = torch.cat(
        [
            dec(h[i]).unsqueeze(-1)
            for i in range(4)
        ],
        dim=-1,
    )
    print(est_source.shape)
    #torch.Size([8, 2, 34, 4]) -> [B, Channel, L, Num_Sources]



