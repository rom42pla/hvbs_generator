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


class MaskedLanguageModelingDataset(Dataset):

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

    def __len__(self) -> int:
        return len(self.data_raw)

    def __getitem__(self, i: int) -> Dict[str, str]:
        item: Dict[str, Union[str, int]] = self.data_raw[i]
        return item

    def load_data(self) -> List[Dict[str, str]]:
        data_raw = []
        for i_item, item in enumerate(self.dataset):
            sentences = sent_tokenize(item)
            for i_sentence, sentence in enumerate(sentences):
                data_raw += [sentence]
        return data_raw


if __name__ == "__main__":
    dataset = SQUADDataset(path=join("..", "datasets", "SQuAD_it-train.json"))
    dataset = NextSentencePredictionDataset(dataset)
    print(len(dataset))
    x = dataset[0]
    pprint(x)
