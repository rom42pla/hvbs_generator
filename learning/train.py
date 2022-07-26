import gc
import logging
from datetime import datetime
from os import makedirs
from os.path import join
from pprint import pformat
from typing import Union, Dict
import torch
from torch.utils.data import Subset
import pytorch_lightning as pl

# sets up the loggers
from learning.arg_parsers.train import get_args
from learning.datasets.objects import RPGObjectDataset
from learning.models.tf import HvbGenerator
from learning.utils import init_logger, set_global_seed, train

init_logger()

# retrieves line arguments
args: Dict[str, Union[bool, str, int, float]] = get_args()
logging.info(f"line args:\n{pformat(args)}")

# sets the random seed
set_global_seed(seed=args['seed'])

# sets up the dataset
dataset = RPGObjectDataset(path=join("data", "oggetti_magici.csv"))

# sets up the model
model: pl.LightningModule = HvbGenerator(
    vocabulary=dataset.tokenizer.get_vocab(),
    embeddings_dim=args['embeddings_dim'],
    num_encoders=args['num_encoders'], num_decoders=args['num_decoders'],
    use_masking=True, mask_perc_min=0.1, mask_perc_max=0.3,
    mix_fourier_with_tokens=True
)

# splits the dataset into training and validation
shuffled_indices = torch.randperm(len(dataset))
dataset_train = Subset(dataset, shuffled_indices[:int(len(dataset) * args['train_set_size'])])
dataset_val = Subset(dataset, shuffled_indices[int(len(dataset) * args['train_set_size']):])

# trains the model
train(
    dataset_train=dataset_train,
    dataset_val=dataset_val,
    model=model,
    **args
)

# frees some memory
del dataset
gc.collect()
