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
from pytorch_lightning import Trainer
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
from torch.utils.data import DataLoader, Subset
from torchaudio import transforms
from transformers import BertTokenizerFast, PreTrainedTokenizerFast, PreTrainedTokenizer
from transformers import AutoTokenizer, AutoModel

from learning.datasets_classes.mlm_dataset import MaskedLanguageModelingDataset
from learning.datasets_classes.nsp_dataset import NextSentencePredictionDataset
from learning.datasets_classes.objects import RPGObjectDataset
from learning.datasets_classes.squad import SQUADDataset
from learning.models.modules import FouriEncoderBlock, FouriEncoder, FouriDecoder
from learning.utils import set_global_seed


class GOH_GPT2(pl.LightningModule):
    def __init__(self,
                 vocabulary: Dict[str, int],
                 start_token: str = "[CLS]",
                 pad_token: str = "[PAD]",
                 end_token: str = "[SEP]",
                 unk_token: str = "[UNK]",
                 mask_token: str = "[MASK]",
                 max_sentence_length: int = 32,

                 embeddings_dim: int = 512,
                 dropout_p: Union[int, float] = 0.25,

                 learning_rate: float = 0.002,

                 use_masking: bool = True,
                 mask_perc_min: float = 0.1,
                 mask_perc_max: float = 0.2,
                 noise_strength: float = 0.1,


                 device: Optional[str] = None):
        super().__init__()

        # metas
        assert isinstance(vocabulary, dict) and all([isinstance(k, str) and isinstance(v, int)
                                                     for k, v in vocabulary.items()])
        self.vocabulary = vocabulary
        assert isinstance(start_token, str) and isinstance(end_token, str) \
               and isinstance(pad_token, str) and isinstance(unk_token, str) and isinstance(mask_token, str) \
               and sorted(list(self.vocabulary.values())) == list(range(len(self.vocabulary)))
        self.start_token, self.end_token = start_token, end_token
        self.pad_token, self.unk_token, self.mask_token = pad_token, unk_token, mask_token
        for i, t in enumerate([self.start_token, self.end_token, self.pad_token, self.unk_token, self.mask_token]):
            if t in self.vocabulary:
                continue
            self.vocabulary.update({
                t: len(self.vocabulary)
            })
        assert sorted(list(self.vocabulary.values())) == list(range(len(self.vocabulary)))
        self.vocabulary_reversed = {v: k for k, v in self.vocabulary.items()}
        assert isinstance(max_sentence_length, int) and max_sentence_length >= 1
        self.max_sentence_length = max_sentence_length
        self.tokenizer = self.get_tokenizer(vocabulary=vocabulary)

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

        assert isinstance(embeddings_dim, int) and embeddings_dim >= 1
        self.embeddings_dim = embeddings_dim

        # optimization
        assert isinstance(learning_rate, float) and learning_rate > 0
        self.learning_rate = learning_rate

        self.gpt2 = AutoModel.from_pretrained("GroNLP/gpt2-small-italian")
        self.gpt2.wte = nn.Embedding(len(self.vocabulary), 768)

        self.reconstruction = nn.Sequential(OrderedDict([
            ("linear", nn.Linear(in_features=768,
                                 out_features=len(self.vocabulary))),
        ]))

        self.float()
        assert device is None or device in {"cuda", "cpu"}
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.to(device)
        self.save_hyperparameters()

    def forward(self, tokens: torch.Tensor):
        tokens = tokens.clone()
        if tokens.device != self.device:
            tokens = tokens.to(self.device)

        with profiler.record_function("embeddings"):
            # adds [CLS] and [SEP] tokens for the decoder
            tokens = torch.cat([
                torch.as_tensor([self.vocabulary[self.start_token]],
                                device=self.device).repeat(tokens.shape[0], 1),
                tokens,
                torch.as_tensor([self.vocabulary[self.end_token]],
                                device=self.device).repeat(tokens.shape[0], 1),
            ], dim=-1)
            # retrieves the embeddings
            # tokens_initial = self.tokens_embedder(tokens)
            # tokens = tokens_initial.clone()
            # if self.training:
            #     tokens = self.add_noise(tokens)
            # # adds positional embeddings
            # tokens = self.add_positional_embeddings_fn(tokens)  # (b, s, d)

        with profiler.record_function("gpt2"):
            tokens = self.gpt2(tokens).last_hidden_state[:, 1:-1] # (b, s, d)

        # print("names_decoded", names_decoded.shape)
        with profiler.record_function("classification"):
            pred_tokens = self.reconstruction(tokens)

        gc.collect()
        return pred_tokens

    def apply_mask(self, x: torch.Tensor):
        x = x.clone()
        mask_rand = torch.rand((x.shape[0], x.shape[1]),
                               dtype=torch.float, device=x.device)
        mask = (mask_rand >= self.mask_perc_min) * (mask_rand <= self.mask_perc_max)
        if x.shape[1] != mask.shape[1]:
            mask = torch.cat(
                [torch.zeros(mask.shape[0], dtype=torch.bool, device=x.device),
                 mask], dim=1)
        x[mask] = self.vocabulary[self.mask_token]
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
        del pe, position, div_term
        gc.collect()
        return x

    def training_step(self, batch, batch_idx):
        return self.step_mlm(batch=batch, batch_idx=batch_idx)

    def validation_step(self, batch, batch_idx):
        return self.step_mlm(batch=batch, batch_idx=batch_idx)

    def step_mlm(self, batch, batch_idx):
        # metas
        phase: str = "train" if self.training is True else "val"
        # tokenizes the data
        self.tokenizer.enable_padding(
            direction="right",
            pad_token=self.pad_token,
            pad_id=self.vocabulary[self.pad_token],
            length=self.max_sentence_length,
        )
        ids = torch.as_tensor([token.ids[:self.max_sentence_length]
                               for token in self.tokenizer.encode_batch(batch)],
                              device=self.device, dtype=torch.long)
        self.tokenizer.no_padding()
        # retrieves the labels
        gt_tokens = ids.clone()
        # applies masking
        # ids = self.apply_mask(ids)
        # del ids_preceding, ids_next, ids_not_next
        # makes the prediction
        pred_tokens = self(ids)
        assert len(ids) == len(pred_tokens) == len(gt_tokens)
        pred_tokens = einops.rearrange(pred_tokens, "b s l -> (b s) l")
        gt_tokens = einops.rearrange(gt_tokens, "b s -> (b s)")
        loss_mlm = F.cross_entropy(input=pred_tokens,
                                   target=gt_tokens,
                                   label_smoothing=0.01 if phase == "train" else 0.0,
                                   ignore_index=self.vocabulary[self.pad_token])
        f1_mlm = torchmetrics.functional.f1_score(preds=pred_tokens,
                                                  target=gt_tokens,
                                                  ignore_index=self.vocabulary[self.pad_token],
                                                  average="micro")
        return {
            "loss": loss_mlm,
            "loss_mlm": loss_mlm,
            "f1_mlm": f1_mlm,
        }

    def step_nsp(self, batch, batch_idx):
        # metas
        phase: str = "train" if self.training is True else "val"
        # tokenizes the data
        for t in [self.tokenizer, self.tokenizer_reconstruction]:
            t.enable_padding(
                direction="right",
                pad_token=self.pad_token,
                pad_id=self.vocabulary[self.pad_token],
                length=self.max_sentence_length,
            )
        ids_preceding, ids_next, ids_not_next = [
            torch.as_tensor([token.ids[:self.max_sentence_length]
                             for token in self.tokenizer.encode_batch(batch[key])],
                            device=self.device, dtype=torch.long)
            for key in ['preceding', 'next', 'not_next']
        ]
        assert ids_preceding.shape == ids_next.shape == ids_not_next.shape
        # ids_preceding_rec, ids_next_rec, ids_not_next_rec = [
        #     torch.as_tensor([token.ids[:self.max_sentence_length]
        #                      for token in self.tokenizer_reconstruction.encode_batch(batch[key])],
        #                     device=self.device, dtype=torch.long)
        #     for key in ['preceding', 'next', 'not_next']
        # ]
        # assert ids_preceding_rec.shape == ids_next_rec.shape == ids_not_next_rec.shape
        # assert ids_preceding_rec.shape == ids_preceding.shape
        for t in [self.tokenizer, self.tokenizer_reconstruction]:
            t.no_padding()
        batch_size = ids_preceding.shape[0]
        # retrieves the labels
        gt_nsp_labels = torch.cat([
            torch.ones(batch_size, device=self.device, dtype=torch.long),
            torch.zeros(batch_size, device=self.device, dtype=torch.long),
        ], dim=0)
        gt_tokens = torch.cat([
            torch.cat([
                ids_preceding,
                torch.as_tensor([self.vocabulary[self.end_token]],
                                device=self.device, dtype=torch.long).repeat(batch_size, 1),
                ids_next,
            ], dim=-1),
            torch.cat([
                ids_preceding,
                torch.as_tensor([self.vocabulary[self.end_token]],
                                device=self.device, dtype=torch.long).repeat(batch_size, 1),
                ids_not_next,
            ], dim=-1)
        ], dim=0)
        # applies masking
        ids_preceding, ids_next, ids_not_next = self.apply_mask(ids_preceding), \
                                                self.apply_mask(ids_next), \
                                                self.apply_mask(ids_not_next)
        tokens = torch.cat([
            torch.cat([
                ids_preceding,
                torch.as_tensor([self.vocabulary[self.end_token]],
                                device=self.device, dtype=torch.long).repeat(batch_size, 1),
                ids_next,
            ], dim=-1),
            torch.cat([
                ids_preceding,
                torch.as_tensor([self.vocabulary[self.end_token]],
                                device=self.device, dtype=torch.long).repeat(batch_size, 1),
                ids_not_next,
            ], dim=-1)
        ], dim=0)
        # del ids_preceding, ids_next, ids_not_next
        # makes the prediction
        pred_tokens, pred_nsp_labels = self(tokens)
        assert len(tokens) == len(pred_tokens) == len(gt_nsp_labels) == len(gt_tokens)
        pred_tokens = einops.rearrange(pred_tokens, "b s l -> (b s) l")
        gt_tokens = einops.rearrange(gt_tokens, "b s -> (b s)")
        loss_nsp = F.cross_entropy(input=pred_nsp_labels,
                                   target=gt_nsp_labels,
                                   label_smoothing=0.1 if phase == "train" else 0.0)
        loss_mlm = F.cross_entropy(input=pred_tokens,
                                   target=gt_tokens,
                                   label_smoothing=0.01 if phase == "train" else 0.0,
                                   ignore_index=self.vocabulary[self.pad_token])
        f1_nsp = torchmetrics.functional.f1_score(preds=pred_nsp_labels,
                                                  target=gt_nsp_labels,
                                                  average="micro")
        f1_mlm = torchmetrics.functional.f1_score(preds=pred_tokens,
                                                  target=gt_tokens,
                                                  ignore_index=self.vocabulary[self.pad_token],
                                                  average="micro")
        del tokens, pred_tokens, gt_nsp_labels, gt_tokens
        gc.collect()
        return {
            "loss": loss_nsp + loss_mlm,
            "loss_nsp": loss_nsp,
            "loss_mlm": loss_mlm,
            "f1_nsp": f1_nsp,
            "f1_mlm": f1_mlm,
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
        # logs
        for metric in outputs[0].keys():
            self.log(f"{metric}_{phase}", torch.stack([e[metric].detach().cpu() for e in outputs]).mean(),
                     prog_bar=True)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer, optimizer_idx):
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.learning_rate)
        return optimizer

    def generate(self, max_length: int = 32):
        tokens = []
        for _ in range(max_length):
            next_token = self.predict_next_token(previous_tokens=tokens)
            if next_token == self.end_token:
                break
            tokens += [next_token]
        # merges the tokens
        generated_string = ""
        for token in tokens:
            if token.startswith("##"):
                generated_string += token[2:]
            else:
                generated_string += f" {token}"
        generated_string = generated_string.strip().capitalize()
        return generated_string

    def predict_next_token(self, previous_tokens: List[str]):
        previous_tokens = [self.start_token] + previous_tokens
        was_training: bool = self.training
        with torch.no_grad():
            self.training = False
            tokens = torch.as_tensor([token.ids
                                      for token in self.tokenizer.encode_batch(previous_tokens)],
                                     device=self.device)
            # retrieves the embeddings
            tokens_initial = self.tokens_embedder(tokens)
            tokens = tokens_initial.clone()
            tokens = self.add_positional_embeddings_fn(tokens)  # (b, s, d)
            # encoder pass
            tokens = self.encoder(tokens)  # (b, s, d)
            # decoder pass
            # pad_embedding = self.tokens_embedder(torch.as_tensor([self.vocabulary[self.pad_token]],
            #                                                      device=self.device))
            # tokens_initial_shifted = torch.cat([tokens_initial[:, 0:1],
            #                                     tokens_initial[:, 2:],
            #                                     pad_embedding.repeat(tokens_initial.shape[0], 1, 1)],
            #                                    dim=1)
            # tokens_initial_shifted = self.add_positional_embeddings_fn(tokens_initial_shifted)
            # tokens = self.decoder(x_encoder=tokens, x_decoder=tokens_initial_shifted)  # (b, s, d)
            pred_next_token_id = self.reconstruction(tokens)[0, -1]
            pred_next_token_id = F.softmax(pred_next_token_id, dim=-1)
            pred_next_token_id = torch.argmax(pred_next_token_id).detach().item()
            pred_next_token = self.classification_bindings[pred_next_token_id]
        if was_training:
            self.training = True
        return pred_next_token

    def get_tokenizer(self, vocabulary: Dict[str, int]) -> Tokenizer:
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenizer: Tokenizer = Tokenizer(WordPiece(vocab=vocabulary,
                                                   unk_token=self.unk_token,
                                                   max_input_chars_per_word=64))
        tokenizer.normalizer = BertNormalizer(
            clean_text=False,
            handle_chinese_chars=False,
            strip_accents=False,
            lowercase=True,
        )
        tokenizer.decoder = decoders.WordPiece()
        return tokenizer


class AddGaussianNoise(nn.Module):
    def __init__(self, strength: float = 0.1):
        super().__init__()
        assert strength >= 0
        self.strength = strength

    def forward(self, x: torch.Tensor):
        noise = torch.normal(mean=torch.zeros_like(x, device=x.device) + x.mean(),
                             std=torch.zeros_like(x, device=x.device) + x.std())
        noise = noise * self.strength
        x = x + noise
        del noise
        gc.collect()
        return x


if __name__ == "__main__":
    set_global_seed(42)
    dataset = RPGObjectDataset(path=join("..", "datasets", "oggetti_magici.csv"),
                               max_length=32)
    tokenizer = BertTokenizerFast(join("..", "datasets_classes", "vocab.txt"), lowercase=True)
    tokens = dataset.get_used_tokens(tokenizer=tokenizer)
    vocabulary = {v: i for i, v in enumerate(tokens)}
    model = GOH_GPT2(embeddings_dim=512, vocabulary=vocabulary,
                         start_token=tokenizer.cls_token,
                         end_token=tokenizer.sep_token,
                         pad_token=tokenizer.pad_token,
                         unk_token=tokenizer.unk_token,
                         use_masking=True,
                         mask_perc_min=0.1, mask_perc_max=0.2)
    model.training = True
    shuffled_indices = torch.randperm(len(dataset))
    objects_dataset_train = MaskedLanguageModelingDataset(Subset(dataset,
                                                                 shuffled_indices[:int(len(dataset) * 0.8):]))
    objects_dataset_val = MaskedLanguageModelingDataset(Subset(dataset,
                                                               shuffled_indices[int(len(dataset) * 0.2):]))
    print(model)
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True, profile_memory=True) as prof:
        dataloader_train = DataLoader(objects_dataset_train, batch_size=16, shuffle=True,
                                      num_workers=os.cpu_count() - 2)
        dataloader_val = DataLoader(objects_dataset_val, batch_size=16, shuffle=False,
                                    num_workers=os.cpu_count() - 2)
        trainer = pl.Trainer(
            gpus=1 if torch.cuda.is_available() else 0,
            precision=32,
            max_epochs=100,
            check_val_every_n_epoch=1,
            logger=False,
            log_every_n_steps=1,
            enable_progress_bar=True,
            enable_model_summary=True,
            enable_checkpointing=False,
            gradient_clip_val=1,
            auto_lr_find=False,
        )
        trainer.fit(model=model, train_dataloaders=dataloader_train, val_dataloaders=dataloader_val)
    print(prof.key_averages(group_by_input_shape=False).table(sort_by="cpu_time", row_limit=16))

    for _ in range(4):
        print(model.generate())
