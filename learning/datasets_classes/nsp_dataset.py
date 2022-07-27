import random
import time
from abc import abstractmethod, ABC
from copy import deepcopy
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename, exists
from pprint import pprint
from typing import Dict, Optional, Union, List, Set, Tuple, Any

import torch
from nltk import sent_tokenize
from torch.utils.data import Dataset, DataLoader, Subset

import os

import scipy.io as sio
import numpy as np
import pickle
import einops
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import BertTokenizerFast, AutoTokenizer
import nltk

from learning.datasets_classes.base_dataset import BaseDataset
from learning.datasets_classes.squad import SQUADDataset


class NextSentencePredictionDataset(Dataset):

    def __init__(
            self,
            dataset: Union[BaseDataset, Subset[BaseDataset]]
    ):
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        super().__init__()
        assert isinstance(dataset, BaseDataset) \
               or (isinstance(dataset, Subset) and isinstance(dataset.dataset, BaseDataset))
        self.dataset = dataset
        self.data_raw = self.load_data()
        self.indices_with_preceding: List[int] = [i_item for i_item, item in enumerate(self.data_raw)
                                                  if item['i_preceding'] is not None]

    def __len__(self) -> int:
        return len(self.indices_with_preceding)

    def __getitem__(self, i: int) -> Dict[str, str]:
        item: Dict[str, Union[str, int]] = self.data_raw[self.indices_with_preceding[i]]
        preceding_item: Dict[str, Union[str, int]] = self.data_raw[item['i_preceding']]
        random_items = random.choices(self.data_raw, k=16)
        random_item: Dict[str, Union[str, int]] = [random_item for random_item in random_items
                                                   if random_item['i_preceding'] != item['i_current']][0]
        x = {
            "preceding": preceding_item['sentence'],
            "next": item['sentence'],
            "not_next": random_item['sentence'],
        }
        return x

    def load_data(self) -> List[Dict[str, str]]:
        data_raw = []
        for i_item, item in enumerate(self.dataset):
            sentences = sent_tokenize(item)
            for i_sentence, sentence in enumerate(sentences):
                data_raw += [{
                    "i_current": len(data_raw),
                    "i_preceding": None if i_sentence == 0 else (len(data_raw) - 1),
                    "sentence": sentence,
                }]
        return data_raw


if __name__ == "__main__":
    dataset = SQUADDataset(path=join("..", "datasets", "SQuAD_it-train.json"))
    dataset = NextSentencePredictionDataset(dataset)
    print(len(dataset))
    x = dataset[0]
    pprint(x)
