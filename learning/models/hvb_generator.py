import gc
import logging
import math
import os
from collections import OrderedDict
from os.path import join
from pprint import pprint
from tokenize import Whitespace
from typing import Union, List, Optional, Dict

import functorch
import torch
import torchvision
from einops.layers.torch import Rearrange
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tokenizers import pre_tokenizers, decoders, Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.processors import BertProcessing
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import einops
import torch.autograd.profiler as profiler
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader
from torchaudio import transforms
from transformers import BertTokenizerFast, PreTrainedTokenizerFast, PreTrainedTokenizer

from learning.datasets_classes.objects import RPGObjectDataset
from learning.datasets_classes.squad import SQUADDataset
from learning.models.modules import FouriEncoderBlock, FouriEncoder, FouriDecoder


class HvbGenerator(pl.LightningModule):
    def __init__(self,
                 vocabulary: Dict[str, int],
                 start_token: str = "[CLS]",
                 pad_token: str = "[PAD]",
                 end_token: str = "[SEP]",
                 unk_token: str = "[UNK]",
                 mask_token: str = "[MASK]",
                 max_sentence_length: int = 32,

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
        assert isinstance(start_token, str) and isinstance(end_token, str) \
               and isinstance(pad_token, str) and isinstance(unk_token, str) and isinstance(mask_token, str)
        self.start_token, self.end_token = start_token, end_token
        self.pad_token, self.unk_token, self.mask_token = pad_token, unk_token, mask_token
        self.vocabulary.update({
            t: len(self.vocabulary) + i
            for i, t in enumerate([self.start_token, self.end_token, self.pad_token, self.unk_token, self.mask_token])
        })
        self.vocabulary_reversed = {v: k for k, v in self.vocabulary.items()}
        self.classification_bindings = {k: i for i, k in enumerate([k for k in self.vocabulary.keys()
                                                                    if k not in {self.start_token, self.mask_token}])}
        assert isinstance(max_sentence_length, int) and max_sentence_length >= 1
        self.max_sentence_length = max_sentence_length
        self.tokenizer = self.get_tokenizer()

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

        self.encoder = FouriEncoder(
            embeddings_dim=self.embeddings_dim,
            num_encoders=self.num_encoders,
            dropout_p=self.dropout_p,
            mix_fourier_with_tokens=self.mix_fourier_with_tokens,
        )
        self.decoder = FouriDecoder(
            embeddings_dim=self.embeddings_dim,
            num_decoders=self.num_decoders,
            dropout_p=self.dropout_p,
            mix_fourier_with_tokens=self.mix_fourier_with_tokens,
        )

        self.classification = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(in_features=self.embeddings_dim,
                                 out_features=len(self.classification_bindings))),
        ]))

        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    def forward(self,
                tokens: torch.Tensor):
        tokens = tokens.clone()
        if tokens.device != self.device:
            tokens = tokens.to(self.device)

        with profiler.record_function("masking"):
            if self.training and self.use_masking:
                mask_rand = torch.rand((tokens.shape[0], tokens.shape[1]),
                                       dtype=torch.float, device=self.device)
                mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
                if tokens.shape[1] != mask.shape[1]:
                    mask = torch.cat(
                        [torch.zeros(mask.shape[0], self.mask_start_index, dtype=torch.bool, device=mask.device),
                         mask], dim=1)
                tokens[mask] = self.vocabulary[self.mask_token]

        with profiler.record_function("embeddings"):
            # adds [CLS] and [SEP] tokens
            tokens = torch.cat([
                torch.as_tensor([self.vocabulary[self.start_token]], device=self.device).repeat(tokens.shape[0], 1),
                tokens,
                torch.as_tensor([self.vocabulary[self.end_token]], device=self.device).repeat(tokens.shape[0], 1),
            ], dim=-1)
            # retrieves the embeddings
            tokens_initial = self.tokens_embedder(tokens)
            tokens = tokens_initial.clone()
            if self.training:
                tokens = self.add_noise(tokens)
            # adds positional embeddings
            tokens = self.add_positional_embeddings_fn(tokens)  # (b, s, d)

        with profiler.record_function("encoder"):
            tokens = self.encoder(tokens)  # (b, s, d)

        # print("names_encoded", names_encoded.shape)
        with profiler.record_function("decoder"):
            tokens_initial = self.add_positional_embeddings_fn(tokens_initial)
            pad_embedding = self.tokens_embedder(torch.as_tensor([self.vocabulary[self.pad_token]],
                                                                 device=self.device))
            tokens_initial_shifted = torch.cat([tokens_initial[:, 0:1],
                                                tokens_initial[:, 2:],
                                                pad_embedding.repeat(tokens_initial.shape[0], 1, 1)],
                                               dim=1)
            tokens = self.decoder(x_encoder=tokens, x_decoder=tokens_initial_shifted)  # (b, s, d)
            tokens = tokens[:, 1:-1]
        # print("names_decoded", names_decoded.shape)
        with profiler.record_function("classification"):
            pred_tokens = self.classification(tokens)

        return pred_tokens

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

    def training_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step(batch=batch, batch_idx=batch_idx)

    def step(self, batch, batch_idx):
        # name of the current phase
        phase: str = "train" if self.training is True else "val"
        # retrieves the data
        if "name" in batch.keys():
            tokens_raw: torch.Tensor = batch["name"]
        elif "context" in batch.keys():
            tokens_raw: torch.Tensor = batch["context"]
        # tokenizes the data
        self.tokenizer.enable_padding(
            direction="right",
            pad_token=self.pad_token,
            pad_id=self.vocabulary[self.pad_token],
            length=self.max_sentence_length,
        )
        tokens = torch.as_tensor([o.ids for o in self.tokenizer.encode_batch(tokens_raw)], device=self.device)
        self.tokenizer.no_padding()
        # makes the prediction
        pred_tokens = self(tokens)
        # computes the metrics
        gt_tokens = torch.cat([
            tokens[:, 1:],
            torch.as_tensor([self.vocabulary[self.pad_token]], device=self.device).repeat(tokens.shape[0], 1)
        ], dim=-1)
        for value in gt_tokens.unique():
            value = value.detach().item()
            classification_binding = self.classification_bindings[self.vocabulary_reversed[value]]
            gt_tokens[gt_tokens == value] = classification_binding
        pred_tokens = einops.rearrange(pred_tokens, "b s l -> (b s) l")
        gt_tokens = einops.rearrange(gt_tokens, "b s -> (b s)")
        loss = F.cross_entropy(input=pred_tokens,
                               target=gt_tokens,
                               label_smoothing=0.1 if phase == "train" else 0.0,
                               ignore_index=self.classification_bindings[self.pad_token])
        f1 = torchmetrics.functional.f1_score(preds=pred_tokens,
                                              target=gt_tokens,
                                              ignore_index=self.classification_bindings[self.pad_token],
                                              average="micro")
        del pred_tokens, gt_tokens
        gc.collect()
        return {
            "loss": loss,
            "f1": f1,
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
        losses: torch.Tensor = torch.stack([e["loss"].detach().cpu() for e in outputs])
        self.log(f"loss_{phase}", losses.mean(), prog_bar=True if phase == "val" else False)
        # f1
        f1s: torch.Tensor = torch.stack([e["f1"].detach().cpu() for e in outputs])
        self.log(f"f1_{phase}", f1s.mean(), prog_bar=True)
        del losses, f1s
        gc.collect()

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate)
        return optimizer

    def generate(self):
        tokens = []
        for _ in range(20):
            with torch.no_grad():
                next_token = self(tokens=torch.as_tensor([
                    self.vocabulary[self.start_token],
                    *tokens,
                    self.vocabulary[self.end_token]
                ], device=self.device).repeat(1, 1))[:, -1]
                next_token = F.softmax(next_token, dim=-1)
                next_token = torch.argmax(next_token, dim=-1)
                next_token = next_token.detach().item()
                if next_token == self.vocabulary["[SEP]"]:
                    break
                tokens += [next_token]
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

    def get_tokenizer(self) -> Tokenizer:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer: Tokenizer = Tokenizer(WordPiece(vocab=self.vocabulary,
                                                   unk_token=self.unk_token,
                                                   max_input_chars_per_word=64))
        tokenizer.normalizer = BertNormalizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True,
        )
        tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        tokenizer.decoder = decoders.WordPiece()
        # self.tokenizer.post_processor = BertProcessing(
        #     ("[SEP]", self.tokenizer.token_to_id("[SEP]")),
        #     ("[CLS]", self.tokenizer.token_to_id("[CLS]")),
        # )
        return tokenizer


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
    dataset = RPGObjectDataset(path=join("..", "datasets", "oggetti_magici.csv"),
                               max_length=32, vocab_path=join("..", "datasets_classes", "vocab.txt"))
    # squad_train = SQUADDataset(path=join("..", "datasets", "SQuAD_it-train.json"),
    #                            vocab_path=join("..", "datasets_classes", "vocab.txt"))
    tokenizer = BertTokenizerFast(join("..", "datasets_classes", "vocab.txt"), lowercase=True)
    tokens = dataset.get_used_tokens(tokenizer=tokenizer)
    vocabulary = {v: i for i, v in enumerate(tokens)}
    model = HvbGenerator(embeddings_dim=128, vocabulary=vocabulary,
                         start_token=tokenizer.cls_token,
                         end_token=tokenizer.sep_token,
                         pad_token=tokenizer.pad_token,
                         unk_token=tokenizer.unk_token,
                         num_encoders=2, num_decoders=2,
                         use_masking=True,
                         mask_perc_min=0.1, mask_perc_max=0.3,
                         mix_fourier_with_tokens=True)
    model.training = True
    print(model)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        dataloader = DataLoader(dataset, batch_size=8)
        for b in dataloader:
            model.training_step(b, 0)
            print(b)
            break
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))

    # with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
    #     dataloader = DataLoader(dataset, batch_size=64)
    #     model.training_step(next(iter(dataloader)), 0)
    # print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=8))
    for _ in range(4):
        print(model.generate())
