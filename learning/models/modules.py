import math
from collections import OrderedDict
from typing import Union

import functorch
import torch
from einops.layers.torch import Rearrange
from torch import nn
import torch.nn.functional as F
import einops


class FouriEncoder(nn.Module):
    def __init__(self,
                 embeddings_dim: int,
                 num_encoders: int = 6,
                 dropout_p: float = 0.1,

                 use_masking: bool = True,
                 mask_perc_min: float = 0.05,
                 mask_perc_max: float = 0.15,
                 mask_start_index: int = 0,

                 add_positional_embeddings: bool = True,
                 mix_fourier_with_tokens: bool = True,
                 ):
        super().__init__()

        # model architecture
        assert isinstance(num_encoders, int) and num_encoders >= 1, \
            f"there must be at least one encoder, not {num_encoders}"
        self.num_encoders = num_encoders
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1, \
            f"embeddings must be greater than 0, not {embeddings_dim}"
        self.embeddings_dim = embeddings_dim
        assert 0 <= dropout_p < 1, \
            f"dropout must be in [0, 1], not {dropout_p}"
        self.dropout_p = dropout_p
        assert isinstance(add_positional_embeddings, bool)
        self.add_positional_embeddings = add_positional_embeddings
        assert isinstance(mix_fourier_with_tokens, bool)
        self.mix_fourier_with_tokens = mix_fourier_with_tokens

        # masking
        assert isinstance(use_masking, bool)
        self.use_masking = use_masking
        if self.use_masking is True:
            assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
            assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
            self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min
            assert isinstance(mask_start_index, int) and mask_start_index >= 0
            self.mask_start_index = mask_start_index
        else:
            self.mask_perc_max, self.mask_perc_min = None, None
            self.mask_start_index = 0

        # special tokens dict
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(["[start]", "[end]", "[mask]"])
        }

        # architecture
        self.tokens_embedder = nn.Embedding(len(self.special_tokens), self.embeddings_dim)
        self.encoder_blocks = nn.Sequential(OrderedDict([*[(f"enc_{i}",
                                                            FouriEncoderBlock(in_features=self.embeddings_dim,
                                                                              mid_features=self.embeddings_dim,
                                                                              out_features=self.embeddings_dim,
                                                                              dropout_p=self.dropout_p,
                                                                              mix_fourier_with_tokens=self.mix_fourier_with_tokens))
                                                           for i in range(self.num_encoders)],
                                                         ("pooler", nn.Linear(in_features=self.embeddings_dim,
                                                                              out_features=self.embeddings_dim)),
                                                         ("act", nn.SELU()),
                                                         ]))

    def forward(self, x: torch.Tensor):
        # prepares the input
        assert x.shape[-1] == self.embeddings_dim
        assert len(x.shape) in {2, 3}
        is_batched = True if len(x.shape) == 3 else False
        if not is_batched:
            x = einops.rearrange(x, "s c -> () s c")
        input_shape = x.shape

        # generates special tokens
        if self.use_masking:
            mask_token = self.tokens_embedder(torch.as_tensor([
                self.special_tokens["[mask]"],
            ], device=x.device))[0]
        if self.add_positional_embeddings:
            start_token, end_token = self.tokens_embedder(torch.as_tensor([
                self.special_tokens["[start]"],
                self.special_tokens["[end]"],
            ], device=x.device))

        # eventually adds masking
        if self.training and self.use_masking:
            mask_rand = torch.rand((x.shape[0], x[:, self.mask_start_index:].shape[1]),
                                   dtype=x.dtype, device=x.device)
            mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
            if x.shape[1] != mask.shape[1]:
                mask = torch.cat(
                    [torch.zeros(mask.shape[0], self.mask_start_index, dtype=torch.bool, device=mask.device),
                     mask], dim=1)
            x[mask] = mask_token

        # adds start and end token
        if self.add_positional_embeddings:
            x = torch.cat([start_token.repeat(x.shape[0], 1, 1),
                           x,
                           end_token.repeat(x.shape[0], 1, 1)], dim=1)
        x[:, self.mask_start_index:] = self.add_positional_embeddings_fn(x[:, self.mask_start_index:])

        # encoders pass
        x = self.encoder_blocks(x)
        if self.add_positional_embeddings:
            x = x[:, 1:-1]

        if not is_batched:
            x = einops.rearrange(x, "b s c -> (b s) c")
        assert x.shape == input_shape, \
            f"output shape {x.shape} is different from input shape {input_shape}"
        return x

    @staticmethod
    def add_positional_embeddings_fn(x: torch.Tensor):
        sequence_length, embeddings_dim = x.shape[-2], x.shape[-1]
        pe = torch.zeros(sequence_length, embeddings_dim, device=x.device)
        position = torch.arange(0, sequence_length, device=x.device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=x.device) *
                              -(math.log(10000.0) / embeddings_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        x = x + pe
        return x


class FouriEncoderBlock(nn.Module):

    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0,
                 mix_fourier_with_tokens: bool = True):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p
        assert isinstance(mix_fourier_with_tokens, bool)
        self.mix_fourier_with_tokens = mix_fourier_with_tokens

        if self.mix_fourier_with_tokens:
            self.fourier_layer = FastFourierTransform()
            self.layer_norm_1 = nn.LayerNorm([in_features, ])
        self.feed_forward_layer = FouriFeedForward(in_features=self.in_features,
                                                   mid_features=self.mid_features,
                                                   out_features=self.out_features,
                                                   dropout_p=dropout_p)
        if self.in_features != self.out_features:
            self.up_projection = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.mid_features),
                nn.SELU(),
                nn.Linear(in_features=self.mid_features, out_features=self.out_features)
            )
        self.layer_norm_2 = nn.LayerNorm([out_features, ])

    def forward(self, x):
        if self.mix_fourier_with_tokens:
            # fourier pass
            x_fourier = self.fourier_layer(x)
            x = self.layer_norm_1(x + x_fourier)
        # fc pass
        x_forwarded = self.feed_forward_layer(x)
        if self.in_features != self.out_features:
            x = self.up_projection(x)
        x = x + x_forwarded
        x = self.layer_norm_2(x)
        return x


class FouriDecoder(nn.Module):
    def __init__(self,
                 embeddings_dim: int,
                 num_decoders: int = 6,
                 dropout_p: float = 0.1,

                 use_masking: bool = True,
                 mask_perc_min: float = 0.05,
                 mask_perc_max: float = 0.15,
                 mask_start_index: int = 0,

                 add_positional_embeddings: bool = True,
                 mix_fourier_with_tokens: bool = True,
                 num_heads: int = 4,
                 ):
        super().__init__()

        # model architecture
        assert isinstance(num_decoders, int) and num_decoders >= 1, \
            f"there must be at least one decoder, not {num_decoders}"
        self.num_decoders: int = num_decoders
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1, \
            f"embeddings must be greater than 0, not {embeddings_dim}"
        self.embeddings_dim = embeddings_dim
        assert 0 <= dropout_p < 1, \
            f"dropout must be in [0, 1], not {dropout_p}"
        self.dropout_p = dropout_p
        assert isinstance(add_positional_embeddings, bool)
        self.add_positional_embeddings = add_positional_embeddings
        assert isinstance(mix_fourier_with_tokens, bool)
        self.mix_fourier_with_tokens = mix_fourier_with_tokens
        assert isinstance(num_heads, int) and num_heads >= 1
        self.num_heads: int = num_heads

        # masking
        assert isinstance(use_masking, bool)
        self.use_masking = use_masking
        if self.use_masking is True:
            assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
            assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
            self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min
            assert isinstance(mask_start_index, int) and mask_start_index >= 0
            self.mask_start_index = mask_start_index
        else:
            self.mask_perc_max, self.mask_perc_min = None, None
            self.mask_start_index = 0

        # special tokens dict
        special_tokens_list = []
        if self.use_masking:
            special_tokens_list += ["[mask]"]
        if self.add_positional_embeddings:
            special_tokens_list += ["[start]", "[end]"]
        self.special_tokens = {
            token: i_token
            for i_token, token in enumerate(special_tokens_list)
        }

        # architecture
        if len(self.special_tokens) >= 1:
            self.tokens_embedder = nn.Embedding(len(self.special_tokens), self.embeddings_dim)
        self.decoder_blocks = nn.ModuleList(
            [FouriDecoderBlock(in_features=self.embeddings_dim,
                               mid_features=self.embeddings_dim,
                               out_features=self.embeddings_dim,
                               dropout_p=self.dropout_p,
                               mix_fourier_with_tokens=self.mix_fourier_with_tokens,
                               num_heads=self.num_heads)
             for _ in range(self.num_decoders)])
        self.postprocessing = nn.Sequential(OrderedDict([
            ("pooler", nn.Linear(in_features=self.embeddings_dim,
                                 out_features=self.embeddings_dim)),
            ("act", nn.SELU()),
        ]))

    def forward(self, x_encoder: torch.Tensor, x_decoder: torch.Tensor):
        # prepares the input
        assert x_encoder.shape[-1] == self.embeddings_dim
        assert x_decoder.shape[-1] == self.embeddings_dim
        assert len(x_encoder.shape) in {2, 3}
        assert len(x_decoder.shape) in {2, 3}
        assert len(x_encoder.shape) == len(x_decoder.shape)
        is_batched = True if len(x_encoder.shape) == 3 else False
        if not is_batched:
            x_encoder = einops.rearrange(x_encoder, "s c -> () s c")
            x_decoder = einops.rearrange(x_decoder, "s c -> () s c")
        input_shape = x_decoder.shape

        # generates special tokens
        if self.use_masking:
            mask_token = self.tokens_embedder(torch.as_tensor([
                self.special_tokens["[mask]"],
            ], device=x_decoder.device))[0]
        if self.add_positional_embeddings:
            start_token, end_token = self.tokens_embedder(torch.as_tensor([
                self.special_tokens["[start]"],
                self.special_tokens["[end]"],
            ], device=x_decoder.device))

        # eventually adds masking
        if self.training and self.use_masking:
            mask_rand = torch.rand((x_decoder.shape[0], x_decoder[:, self.mask_start_index:].shape[1]),
                                   dtype=x_decoder.dtype, device=x_decoder.device)
            mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
            if x_decoder.shape[1] != mask.shape[1]:
                mask = torch.cat(
                    [torch.zeros(mask.shape[0], self.mask_start_index, dtype=torch.bool, device=mask.device),
                     mask], dim=1)
            x_decoder[mask] = mask_token

        # adds start and end token and eventually adds positional embeddings
        if self.add_positional_embeddings:
            x_decoder = torch.cat([start_token.repeat(x_decoder.shape[0], 1, 1),
                                   x_decoder,
                                   end_token.repeat(x_decoder.shape[0], 1, 1)], dim=1)
        x_decoder[:, self.mask_start_index:] = self.add_positional_embeddings_fn(x_decoder[:, self.mask_start_index:])

        # decoders pass
        for decoder_block in self.decoder_blocks:
            x_decoder = decoder_block(x_encoder=x_encoder, x_decoder=x_decoder)
        if self.add_positional_embeddings:
            x_decoder = x_decoder[:, 1:-1]
        # x_decoder = self.postprocessing(x_decoder)

        if not is_batched:
            x_decoder = einops.rearrange(x_decoder, "b s c -> (b s) c")
        assert x_decoder.shape == input_shape, \
            f"output shape {x_decoder.shape} is different from input shape {input_shape}"
        return x_decoder

    @staticmethod
    def add_positional_embeddings_fn(x: torch.Tensor):
        sequence_length, embeddings_dim = x.shape[-2], x.shape[-1]
        pe = torch.zeros(sequence_length, embeddings_dim, device=x.device)
        position = torch.arange(0, sequence_length, device=x.device).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, embeddings_dim, 2, dtype=torch.float, device=x.device) *
                              -(math.log(10000.0) / embeddings_dim)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        x = x + pe
        return x


class FouriDecoderBlock(nn.Module):

    def __init__(
            self,
            in_features: int,
            mid_features: int,
            out_features: int,
            dropout_p: Union[int, float] = 0,
            mix_fourier_with_tokens: bool = True,
            num_heads: int = 4,
    ):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features: int = in_features
        self.mid_features: int = mid_features
        self.out_features: int = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p: float = dropout_p
        assert isinstance(mix_fourier_with_tokens, bool)
        self.mix_fourier_with_tokens: bool = mix_fourier_with_tokens
        assert isinstance(num_heads, int) and num_heads >= 1
        self.num_heads: int = num_heads

        if self.mix_fourier_with_tokens:
            self.fourier_layer = FastFourierTransform()
            self.layer_norm_1 = nn.LayerNorm([in_features, ])

        self.attention = LinearMultiheadAttention(embeddings_dim=self.in_features,
                                                  num_heads=self.num_heads,
                                                  dropout_p=self.dropout_p)
        self.layer_norm_2 = nn.LayerNorm([in_features, ])
        self.feed_forward_layer = FouriFeedForward(in_features=self.in_features,
                                                   mid_features=self.mid_features,
                                                   out_features=self.out_features,
                                                   dropout_p=dropout_p)
        if self.in_features != self.out_features:
            self.up_projection = nn.Sequential(
                nn.Linear(in_features=self.in_features, out_features=self.mid_features),
                nn.SELU(),
                nn.Linear(in_features=self.mid_features, out_features=self.out_features)
            )
        self.layer_norm_3 = nn.LayerNorm([out_features, ])

    def forward(self, x_encoder, x_decoder):
        if self.mix_fourier_with_tokens:
            # fourier pass
            x_decoder_fourier = self.fourier_layer(x_decoder)
            x_decoder = self.layer_norm_1(x_decoder + x_decoder_fourier)
        # mixing with encoder's output
        attentions = self.attention(v=x_encoder, k=x_encoder, q=x_decoder)
        x = self.layer_norm_2(x_decoder + attentions)
        # fc pass
        x_forwarded = self.feed_forward_layer(x)
        if self.in_features != self.out_features:
            x = self.up_projection(x)
        x = x + x_forwarded
        x = self.layer_norm_3(x)
        return x


class LinearMultiheadAttention(nn.Module):
    def __init__(
            self,
            embeddings_dim: int,
            num_heads: int,
            dropout_p: float = 0.0,
    ):
        super().__init__()
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1
        self.embeddings_dim: int = embeddings_dim
        assert isinstance(num_heads, int) and num_heads >= 1
        self.num_heads: int = num_heads
        assert 0 <= dropout_p < 1
        self.dropout_p: float = float(dropout_p)

        self.q_linears = nn.ModuleList([
            nn.Linear(in_features=self.embeddings_dim, out_features=self.embeddings_dim)
            for _ in range(self.num_heads)
        ])
        self.k_linears = nn.ModuleList([
            nn.Linear(in_features=self.embeddings_dim, out_features=self.embeddings_dim)
            for _ in range(self.num_heads)
        ])
        self.v_linears = nn.ModuleList([
            nn.Linear(in_features=self.embeddings_dim, out_features=self.embeddings_dim)
            for _ in range(self.num_heads)
        ])
        self.out_reshaper = nn.Sequential(
            Rearrange("b h s d -> b s (d h)"),
            # nn.AdaptiveMaxPool2d(output_size=(self.embeddings_dim, 1)),
            nn.Linear(in_features=self.embeddings_dim * self.num_heads, out_features=self.embeddings_dim),
            # Rearrange("b s d h -> b s (d h)")
        )

    def forward(self, q, k, v):
        # obtains the embeddings for query, key and values
        qs = torch.stack([
            net(q) for net in self.q_linears
        ], dim=1)  # (b h s d)
        ks = torch.stack([
            net(k) for net in self.k_linears
        ], dim=1)  # (b h s d)
        vs = torch.stack([
            net(v) for net in self.v_linears
        ], dim=1)  # (b h s d)
        # normalizes qs and ks
        qs = F.softmax(qs, dim=1)
        ks = F.softmax(ks, dim=1)
        # gets global contexts
        global_contexts = torch.matmul(ks.mT, vs)
        # gets the output
        out = torch.matmul(qs, global_contexts)
        # reshapes the output
        out = self.out_reshaper(out)
        return out


class FastFourierTransform(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = functorch.vmap(torch.fft.fftn)(x).real
        return x


class FouriFeedForward(nn.Module):
    def __init__(self, in_features: int, mid_features: int, out_features: int,
                 dropout_p: Union[int, float] = 0.1):
        super().__init__()
        assert isinstance(in_features, int) and in_features >= 1
        assert isinstance(mid_features, int) and mid_features >= 1
        assert isinstance(out_features, int) and out_features >= 1
        self.in_features = in_features
        self.mid_features = mid_features
        self.out_features = out_features
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p

        self.linear_1 = nn.Linear(self.in_features, self.mid_features)
        # self.linear_1 = nn.Sequential(
        #     Rearrange("b s c -> b c s"),
        #     nn.Conv1d(in_channels=self.in_features, out_channels=self.mid_features,
        #               kernel_size=7, stride=1, padding=3),
        #     Rearrange("b c s -> b s c"),
        # )
        self.activation = nn.SELU()
        self.linear_2 = nn.Linear(self.mid_features, self.out_features)
        # self.linear_2 = nn.Sequential(
        #     Rearrange("b s c -> b c s"),
        #     nn.Conv1d(in_channels=self.mid_features, out_channels=self.out_features,
        #               kernel_size=5, stride=1, padding=2),
        #     Rearrange("b c s -> b s c"),
        # )
        self.dropout = nn.AlphaDropout(p=self.dropout_p)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x = self.dropout(x)
        return x


if __name__ == "__main__":
    embeddings_dim, batch_size, sampling_rate, seconds = 512, 512, 128, 1
    batch = {
        "eegs": torch.randn(batch_size, seconds * sampling_rate, embeddings_dim, dtype=torch.float32),
        "labels": torch.ones(batch_size, 6, dtype=torch.long),
        "sampling_rates": torch.zeros(batch_size, dtype=torch.long) + sampling_rate,
    }
    batch_target = {
        "eegs": torch.randn(batch_size, 4, embeddings_dim, dtype=torch.float32),
    }
    encoder = FouriEncoder(embeddings_dim=embeddings_dim,
                           num_encoders=2, use_masking=True,
                           mask_perc_min=0.1, mask_perc_max=0.3)
    print(encoder)
    print("input before encoder", batch["eegs"].shape)
    out = encoder(batch["eegs"])
    print("output after encoder", out.shape)

    decoder_block = FouriDecoderBlock(in_features=embeddings_dim,
                                      mid_features=embeddings_dim,
                                      out_features=embeddings_dim,
                                      num_heads=4)
    print(decoder_block)
    print("input before decoder block", batch_target["eegs"].shape)
    out = decoder_block(x_encoder=batch["eegs"], x_decoder=batch_target["eegs"])
    print("output after decoder block", out.shape)

    decoder = FouriDecoder(embeddings_dim=embeddings_dim,
                           num_decoders=2, use_masking=True,
                           mask_perc_min=0.1, mask_perc_max=0.3)
    print(decoder)
    print("input before decoder", batch_target["eegs"].shape)
    out = decoder(batch["eegs"], batch_target["eegs"])
    print("output after decoder", out.shape)
