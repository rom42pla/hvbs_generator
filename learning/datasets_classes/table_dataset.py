import time
from abc import abstractmethod, ABC
from copy import deepcopy
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename, exists
from pprint import pprint
from typing import Dict, Optional, Union, List, Set, Tuple, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import os

import scipy.io as sio
import numpy as np
import pickle
import einops
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizerFast, AutoTokenizer


class RPGTableDataset(Dataset, ABC):

    def __init__(
            self,
            path: str,
            vocab_path: str,
            max_length: int = 512,
    ):
        super().__init__()

        assert exists(path)
        self.path: str = path
        assert exists(vocab_path)
        self.vocab_path: str = vocab_path
        self.data_raw: List[Dict[str, str]] = self.load_data()
        self.tokenizer = BertTokenizerFast(self.vocab_path, lowercase=True)

        # tokenizer = BertWordPieceTokenizer("bert-base-uncased-vocab.txt", lowercase=True)
        assert isinstance(max_length, int) and max_length >= 1
        self.max_length: int = max_length
        # self.setup_data()

    def __len__(self) -> int:
        return len(self.data_raw)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        item_encoded = {}
        for key, sentence in self.data_raw[i].items():
            sentence_ids = torch.as_tensor(
                self.tokenizer.encode(sentence,
                                      padding="max_length", max_length=self.max_length, truncation=True),
                dtype=torch.long)
            # sentence_tokenized = self.tokenizer.tokenize(sentence, add_special_tokens=True)
            # assert len(sentence_tokenized) == len(sentence_ids)
            item_encoded[key] = sentence_ids
        return item_encoded

    @abstractmethod
    def load_data(self) -> List[Dict[str, str]]:
        pass

    def setup_data(self) -> None:
        for item in self.data_raw:
            item_encoded = {}
            for key, sentence in item.items():
                sentence_ids = self.tokenizer.encode(sentence)
                # sentence_tokenized = [self.tokenizer.cls_token] + \
                #                      self.tokenizer.tokenize(sentence) + \
                #                      [self.tokenizer.sep_token]
                # assert len(sentence_tokenized) == len(sentence_ids)
                item_encoded[key] = sentence_ids
            self.data_encoded += [item_encoded]
        assert len(self.data_raw) == len(self.data_encoded)
