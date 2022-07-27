import gc
import logging
from copy import deepcopy
from datetime import datetime
from os import makedirs
from os.path import join
from pprint import pformat, pprint
from typing import Union, Dict
import torch
from torch.utils.data import Subset
import pytorch_lightning as pl

# sets up the loggers
from transformers import BertTokenizerFast

from learning.arg_parsers.train import get_args
from learning.datasets_classes.mlm_dataset import MaskedLanguageModelingDataset
from learning.datasets_classes.nsp_dataset import NextSentencePredictionDataset
from learning.datasets_classes.objects import RPGObjectDataset
from learning.datasets_classes.squad import SQUADDataset
from learning.models.goh_gpt2 import GOH_GPT2
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
                           max_length=args['max_sentences_length'])
squad_test = SQUADDataset(path=join("learning", "datasets", "SQuAD_it-test.json"),
                          max_length=args['max_sentences_length'])
objects_dataset = RPGObjectDataset(path=join("learning", "datasets", "oggetti_magici.csv"),
                                   max_length=args['max_sentences_length'])
logging.info(f"datasets loaded")

tokenizer = BertTokenizerFast(join("learning", "datasets_classes", "vocab.txt"), lowercase=True)
tokens = set()
for dataset in [squad_train, objects_dataset]:
    tokens = tokens.union(set(dataset.get_used_tokens(tokenizer=tokenizer)))
vocabulary = {v: i for i, v in enumerate(tokens)}

# sets up the model
# model: pl.LightningModule = HvbGenerator(
#     vocabulary=vocabulary,
#     start_token=tokenizer.cls_token,
#     end_token=tokenizer.sep_token,
#     pad_token=tokenizer.pad_token,
#     unk_token=tokenizer.unk_token,
#     embeddings_dim=args['embeddings_dim'],
#     num_encoders=args['num_encoders'], num_decoders=args['num_decoders'],
#     use_masking=True, mask_perc_min=0.2, mask_perc_max=0.3,
#     noise_strength=args['noise_strength'], dropout_p=args['dropout_p'],
#     mix_fourier_with_tokens=True,
# )
model: GOH_GPT2 = GOH_GPT2(
    vocabulary=vocabulary,
    start_token=tokenizer.cls_token,
    end_token=tokenizer.sep_token,
    pad_token=tokenizer.pad_token,
    unk_token=tokenizer.unk_token,
    num_layers=args['num_layers'],
    num_heads=args['num_heads'],
    embeddings_dim=args['embeddings_dim'],
    use_masking=True, mask_perc_min=0.2, mask_perc_max=0.3,
    noise_strength=args['noise_strength'], dropout_p=args['dropout_p'],
)
initial_weights = deepcopy(model.state_dict().__str__())

# pre-trains the model
# train(
#     dataset_train=NextSentencePredictionDataset(squad_train),
#     dataset_val=NextSentencePredictionDataset(squad_test),
#     model=model,
#     **args
# )
# assert initial_weights != model.state_dict().__str__(), \
#     f"model not updating"
# for _ in range(8):
#     print(model.generate())

initial_weights = deepcopy(model.state_dict().__str__())
shuffled_indices = torch.randperm(len(objects_dataset))
objects_dataset_train = Subset(objects_dataset, shuffled_indices[:int(len(objects_dataset) * args['train_set_size'])])
objects_dataset_val = Subset(objects_dataset, shuffled_indices[int(len(objects_dataset) * args['train_set_size']):])

# finetune the model
args['learning_rate'] /= 2
train(
    dataset_train=MaskedLanguageModelingDataset(objects_dataset_train),
    dataset_val=MaskedLanguageModelingDataset(objects_dataset_val),
    model=model,
    **args
)
assert initial_weights != model.state_dict().__str__(), \
    f"model not updating"

for sentence in model.generate(times=4, starting_string="anello"):
    print(sentence)

# frees some memory
del objects_dataset
gc.collect()

# print(logs)
