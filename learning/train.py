import gc
import logging
from copy import deepcopy
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
from learning.datasets_classes.objects import RPGObjectDataset
from learning.datasets_classes.squad import SQUADDataset
from learning.models.hvb_generator import HvbGenerator
from learning.utils import init_logger, set_global_seed, train

init_logger()

# retrieves line arguments
args: Dict[str, Union[bool, str, int, float]] = get_args()
logging.info(f"line args:\n{pformat(args)}")

# sets the random seed
set_global_seed(seed=args['seed'])

# sets up the datasets
squad_train = SQUADDataset(path=join("learning", "datasets", "SQuAD_it-train.json"),
                           vocab_path=join("learning", "datasets_classes", "vocab.txt"),)
squad_test = SQUADDataset(path=join("learning", "datasets", "SQuAD_it-test.json"),
                          vocab_path=join("learning", "datasets_classes", "vocab.txt"),)
logging.info(f"SQUAD dataset loaded")
#
# sets up the model
model: pl.LightningModule = HvbGenerator(
    vocabulary=squad_train.tokenizer.get_vocab(),
    embeddings_dim=args['embeddings_dim'],
    num_encoders=args['num_encoders'], num_decoders=args['num_decoders'],
    use_masking=True, mask_perc_min=0.2, mask_perc_max=0.3,
    noise_strength=args['noise_strength'], dropout_p=args['dropout_p'],
    mix_fourier_with_tokens=True,
)
# initial_weights = deepcopy(model.state_dict().__str__())
#
# # pre-trains the model
# train(
#     dataset_train=squad_train,
#     dataset_val=squad_test,
#     model=model,
#     **args
# )
# assert initial_weights != model.state_dict().__str__(), \
#     f"model not updating"

initial_weights = deepcopy(model.state_dict().__str__())
dataset = RPGObjectDataset(path=join("learning", "datasets", "oggetti_magici.csv"),
                           vocab_path=join("learning", "datasets_classes", "vocab.txt"),
                           max_length=args['max_sentences_length'])
shuffled_indices = torch.randperm(len(dataset))
dataset_train = Subset(dataset, shuffled_indices[:int(len(dataset) * args['train_set_size'])])
dataset_val = Subset(dataset, shuffled_indices[int(len(dataset) * args['train_set_size']):])
logging.info(f"Hvb dataset loaded")
# finetune the model
args['learning_rate'] *= 0.1
train(
    dataset_train=dataset_train,
    dataset_val=dataset_val,
    model=model,
    **args
)
assert initial_weights != model.state_dict().__str__(), \
    f"model not updating"

for _ in range(8):
    print(model.generate())

# frees some memory
del dataset
gc.collect()

# print(logs)
