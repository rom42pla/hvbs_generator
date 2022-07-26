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

from learning.datasets_classes.table_dataset import RPGTableDataset


class RPGObjectDataset(RPGTableDataset):

    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)

    def load_data(self) -> List[Dict[str, str]]:
        data_df = pd.read_csv(self.path)
        data = []
        for name, description, rarity in zip(data_df["name"].tolist(),
                                             data_df["description"].tolist(),
                                             data_df["rarity"].tolist()):
            data += [{
                "name": name,
                "description": description,
                "rarity": rarity,
            }]
        # assert all([len(v) == len(data["names"]) for k, v in data.items()])
        return data


if __name__ == "__main__":
    dataset = RPGObjectDataset(path=join("..", "..", "data", "oggetti_magici.csv"))
    x = dataset[0]
    dataloader = DataLoader(dataset, batch_size=64)
    print(dataset.tokenizer.pad_token)
