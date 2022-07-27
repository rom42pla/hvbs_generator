import json
import time
from abc import abstractmethod, ABC
from copy import deepcopy
from multiprocessing import Pool
from os.path import isdir, join, splitext, basename
from pprint import pprint
from typing import Dict, Optional, Union, List, Set, Tuple

import pandas as pd
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

from learning.datasets_classes.base_dataset import BaseDataset


class SQUADDataset(BaseDataset):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def load_data(self) -> List[Dict[str, str]]:
        with open(self.path, "r") as fp:
            data_raw = json.load(fp)["data"]
        data: List[Dict[str, str]] = []
        for i_page, page in enumerate(data_raw):
            data += [" ".join([paragraph["context"]
                               for paragraph in data_raw[i_page]["paragraphs"]])]
        return data


if __name__ == "__main__":
    dataset = SQUADDataset(path=join("..", "datasets", "SQuAD_it-train.json"))
    x = dataset[0]
    # print(len(dataset))
    for x in dataset:
        from nltk.tokenize import sent_tokenize

        print(sent_tokenize(x))

        # pprint(x)
