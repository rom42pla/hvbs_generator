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
import nltk


class BaseDataset(Dataset, ABC):

    def __init__(
            self,
            path: str,
            # vocab_path: str,
            max_length: int = 512,
    ):
        super().__init__()

        assert exists(path)
        self.path: str = path
        # assert exists(vocab_path)
        # self.vocab_path: str = vocab_path
        self.data_raw: List[Dict[str, str]] = self.load_data()
        assert isinstance(max_length, int) and max_length >= 1
        self.max_length: int = max_length

    def __len__(self) -> int:
        return len(self.data_raw)

    def __getitem__(self, i: int) -> Dict[str, str]:
        return self.data_raw[i]

    @abstractmethod
    def load_data(self) -> List[Dict[str, str]]:
        pass

    def get_used_tokens(self, tokenizer: BertTokenizerFast) -> List[str]:
        tokens = set()
        for item in self.data_raw:
            tokens = tokens.union(set(tokenizer.tokenize(item)))
        return list(tokens)
