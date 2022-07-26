import gc
import math
from collections import OrderedDict
from os.path import join
from typing import Union, List, Optional, Dict

import functorch
import torch
import torchvision
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops
import torch.autograd.profiler as profiler
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from torchaudio import transforms

from learning.datasets_classes.objects import RPGObjectDataset
from learning.models.modules import FouriEncoderBlock, FouriEncoder, FouriDecoder


class HvbGenerator(pl.LightningModule):
    def __init__(self,
                 vocabulary: Dict[str, int],
                 start_token: str = "[CLS]",
                 pad_token: str = "[PAD]",
                 end_token: str = "[SEP]",

                 embeddings_dim: int = 512,
                 num_encoders: int = 1,
                 num_decoders: int = 1,
                 dropout_p: Union[int, float] = 0.25,

                 learning_rate: float = 0.002,

                 use_masking: bool = True,
                 mask_perc_min: float = 0.2,
                 mask_perc_max: float = 0.3,
                 noise_strength: float = 0.1,

                 mix_fourier_with_tokens: bool = True,

                 device: Optional[str] = None):
        super().__init__()

        # metas
        assert isinstance(vocabulary, dict) and all([isinstance(k, str) and isinstance(v, int)
                                                     for k, v in vocabulary.items()])
        self.vocabulary = vocabulary
        self.vocabulary_reversed = {v: k for k, v in self.vocabulary.items()}
        assert isinstance(start_token, str) and isinstance(end_token, str) and isinstance(pad_token, str)
        self.start_token, self.end_token, self.pad_token = start_token, end_token, pad_token
        # self.vocabulary_predictable = {k: i for i, (k, _) in enumerate(self.vocabulary.items())
        #                                if k != self.pad_token}
        # self.vocabulary_predictable_reversed = {v: k for k, v in self.vocabulary_predictable.items()}

        # preprocessing
        assert isinstance(mix_fourier_with_tokens, bool)
        self.mix_fourier_with_tokens = mix_fourier_with_tokens

        # regularization
        assert isinstance(use_masking, bool)
        self.use_masking = use_masking
        if self.use_masking is True:
            assert isinstance(mask_perc_min, float) and 0 <= mask_perc_min < 1
            assert isinstance(mask_perc_max, float) and 0 <= mask_perc_max < 1 and mask_perc_max >= mask_perc_min
            self.mask_perc_max, self.mask_perc_min = mask_perc_max, mask_perc_min
        else:
            self.mask_perc_max, self.mask_perc_min = None, None
        assert 0 <= dropout_p < 1
        self.dropout_p = dropout_p
        assert noise_strength >= 0
        self.noise_strength = noise_strength

        # model architecture
        assert isinstance(num_encoders, int) and num_encoders >= 1
        self.num_encoders: int = num_encoders
        assert isinstance(num_decoders, int) and num_decoders >= 1
        self.num_decoders = num_decoders
        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1
        self.embeddings_dim = embeddings_dim

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        self.tokens_embedder = nn.Embedding(len(self.vocabulary), self.embeddings_dim)
        self.add_noise = nn.Sequential(
            AddGaussianNoise(strength=self.noise_strength)
        )

        self.encoder = FouriEncoder(embeddings_dim=self.embeddings_dim,
                                    num_encoders=self.num_encoders,
                                    dropout_p=self.dropout_p,
                                    use_masking=self.use_masking,
                                    mask_perc_min=self.mask_perc_min,
                                    mask_perc_max=self.mask_perc_max,
                                    mask_start_index=0,
                                    add_positional_embeddings=False,
                                    mix_fourier_with_tokens=self.mix_fourier_with_tokens,
                                    )
        self.decoder = FouriDecoder(embeddings_dim=self.embeddings_dim,
                                    num_decoders=self.num_decoders,
                                    dropout_p=self.dropout_p,
                                    use_masking=self.use_masking,
                                    mask_perc_min=self.mask_perc_min,
                                    mask_perc_max=self.mask_perc_max,
                                    mask_start_index=0,
                                    add_positional_embeddings=False,
                                    mix_fourier_with_tokens=self.mix_fourier_with_tokens,
                                    )

        self.classification = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(in_features=self.embeddings_dim,
                                 out_features=len(self.vocabulary))),
        ]))

        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    def forward(self,
                names: torch.Tensor):
        if names.device != self.device:
            names = names.to(self.device)
        # print("names", names.shape)
        with profiler.record_function("embeddings"):
            names_embeddings = self.tokens_embedder(names)
            if self.training:
                names_embeddings = self.add_noise(names_embeddings)
        # print("names_embeddings", names_embeddings.shape)
        with profiler.record_function("encoder"):
            names_encoded = self.encoder(names_embeddings)
        # print("names_encoded", names_encoded.shape)
        with profiler.record_function("decoder"):
            pad_embedding = self.tokens_embedder(torch.as_tensor([self.vocabulary["[PAD]"]],
                                                                 device=self.device))
            names_embeddings_shifted = torch.cat([names_embeddings[:, 0:1],
                                                  names_embeddings[:, 2:],
                                                  pad_embedding.repeat(names_embeddings.shape[0], 1, 1)],
                                                 dim=1)
            names_decoded = self.decoder(x_encoder=names_encoded, x_decoder=names_embeddings_shifted)
        # print("names_decoded", names_decoded.shape)
        with profiler.record_function("classification"):
            pred_tokens = self.classification(names_decoded)

        return pred_tokens

    def training_step(self, batch, batch_idx):
        names: torch.Tensor = batch["name"]
        pred_tokens = self(names)
        gt_tokens = torch.cat([
            names[:, 1:],
            torch.as_tensor([self.vocabulary["[PAD]"]], device=self.device).repeat(names.shape[0], 1)
        ], dim=-1)
        loss = F.cross_entropy(input=einops.rearrange(pred_tokens, "b s l -> (b s) l"),
                               target=einops.rearrange(gt_tokens, "b s -> (b s)"),
                               label_smoothing=0.1,
                               ignore_index=self.vocabulary[self.pad_token])
        return {
            "loss": loss,
            "gt_tokens": gt_tokens,
            "pred_tokens": pred_tokens,
        }

    def validation_step(self, batch, batch_idx):
        names: torch.Tensor = batch["name"]  # (b s)
        pred_tokens = self(names)
        gt_tokens = torch.cat([
            names[:, 1:],
            torch.as_tensor([self.vocabulary["[PAD]"]], device=self.device).repeat(names.shape[0], 1)
        ], dim=-1)
        loss = F.cross_entropy(input=einops.rearrange(pred_tokens, "b s l -> (b s) l"),
                               target=einops.rearrange(gt_tokens, "b s -> (b s)"),
                               ignore_index=self.vocabulary[self.pad_token])
        return {
            "loss": loss,
            "gt_tokens": gt_tokens,
            "pred_tokens": pred_tokens,
        }

    def training_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        self.log_stats(outputs)
        del outputs
        gc.collect()

    def validation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]]):
        self.log_stats(outputs)
        del outputs
        gc.collect()

    def log_stats(self, outputs: List[Dict[str, torch.Tensor]]):
        # name of the current phase
        phase: str = "train" if self.training is True else "val"
        # loss
        losses: torch.Tensor = torch.stack([e["loss"] for e in outputs])
        self.log(f"loss_{phase}", losses.mean(), prog_bar=True if phase == "val" else False)
        # f1
        gt_tokens, pred_tokens = torch.cat([e["gt_tokens"] for e in outputs], dim=0), \
                                 torch.cat([e["pred_tokens"] for e in outputs], dim=0)
        f1 = torchmetrics.functional.f1_score(preds=einops.rearrange(pred_tokens, "b s l -> (b s) l"),
                                              target=einops.rearrange(gt_tokens, "b s -> (b s)"),
                                              ignore_index=self.vocabulary[self.pad_token],
                                              average="micro")
        self.log(f"f1_{phase}", f1, prog_bar=True)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate)
        return optimizer

    def generate(self):
        tokens = torch.as_tensor(
            [self.vocabulary["[CLS]"]],
            device=self.device
        ).repeat(1, 1)
        for _ in range(20):
            next_token = self(tokens)[:, -1]
            next_token = F.softmax(next_token, dim=-1)
            next_token = torch.argmax(next_token, dim=-1)
            if next_token.detach().item() == self.vocabulary["[SEP]"]:
                break
            tokens = torch.cat([tokens, next_token.repeat(1, 1)], dim=-1)
        tokens = tokens.squeeze(0)[1:].detach().tolist()
        tokens = [self.vocabulary_reversed[token_id]
                  for token_id in tokens]
        # merges the tokens
        generated_string = ""
        for token in tokens:
            if token.startswith("##"):
                generated_string += token[2:]
            else:
                generated_string += f" {token}"
        generated_string = generated_string.strip().capitalize()
        return generated_string


class AddGaussianNoise(nn.Module):
    def __init__(self, strength: float = 0.1):
        super().__init__()
        assert strength >= 0
        self.strength = strength

    def forward(self, x: torch.Tensor):
        noise = torch.normal(mean=torch.zeros_like(x, device=x.device, requires_grad=False) + x.mean(),
                             std=torch.zeros_like(x, device=x.device, requires_grad=False) + x.std())
        noise = noise * self.strength
        return x + noise


if __name__ == "__main__":
    dataset = RPGObjectDataset(path=join("..", "..", "data", "oggetti_magici.csv"),
                               max_length=32)
    model = HvbGenerator(embeddings_dim=128, vocabulary=dataset.tokenizer.get_vocab(),
                         num_encoders=1, num_decoders=1,
                         use_masking=True,
                         mask_perc_min=0.1, mask_perc_max=0.3,
                         mix_fourier_with_tokens=True)
    dataloader = DataLoader(dataset, batch_size=8)
    model.training = True
    print(model)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        model.training_step(next(iter(dataloader)), 0)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))
    for _ in range(4):
        print(model.generate())
