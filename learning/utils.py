import gc
import json
import logging
import os
import random
import re
import warnings
from copy import deepcopy
from os.path import join, isdir, exists
from typing import Dict, Any, List, Union, Optional

import numpy as np
import pandas as pd

import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks import EarlyStopping, RichProgressBar, StochasticWeightAveraging
from pytorch_lightning.callbacks.progress.rich_progress import RichProgressBarTheme
from pytorch_lightning.utilities.warnings import LightningDeprecationWarning
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl

from learning.loggers.hvb_logger import HvbLogger


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def init_logger() -> None:
    pd.set_option('display.max_columns', None)
    warnings.filterwarnings("ignore", category=LightningDeprecationWarning)
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    logging.basicConfig(
        format='\x1b[42m\x1b[30m[%(asctime)s, %(levelname)s]\x1b[0m %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


def init_callbacks(swa: bool = False) -> List[Callback]:
    callbacks: List[Callback] = [
        EarlyStopping(monitor="loss_val", mode="min", min_delta=1e-3, patience=5,
                      verbose=False, check_on_train_epoch_end=False, strict=True),
        RichProgressBar(
            theme=RichProgressBarTheme(
                description="green_yellow",
                progress_bar="green1",
                progress_bar_finished="green1",
                progress_bar_pulse="#6206E0",
                batch_progress="green_yellow",
                time="grey82",
                processing_speed="grey82",
                metrics="grey82",
            )
        ),
    ]
    if swa:
        callbacks += [
            StochasticWeightAveraging(),
        ]
    return callbacks


def save_to_json(object: Any, path: str) -> None:
    with open(path, 'w') as fp:
        json.dump(object, fp, indent=4)


def train(
        dataset_train: Dataset,
        dataset_val: Dataset,
        model: pl.LightningModule,
        batch_size: int = 64,
        max_epochs: int = 1000,
        precision: int = 32,
        auto_lr_finder: bool = False,
        gradient_clipping: bool = False,
        stochastic_weight_average: bool = False,
        **kwargs,
) -> pd.DataFrame:
    initial_weights = deepcopy(model.state_dict().__str__())

    dataloader_train: DataLoader = DataLoader(dataset_train,
                                              batch_size=batch_size, shuffle=True,
                                              num_workers=os.cpu_count() - 1,
                                              pin_memory=True if torch.cuda.is_available() else False)
    dataloader_val: DataLoader = DataLoader(dataset_val,
                                            batch_size=batch_size, shuffle=False,
                                            num_workers=os.cpu_count() - 1,
                                            pin_memory=True if torch.cuda.is_available() else False)

    # frees some memory
    gc.collect()

    # initializes the trainer
    trainer = pl.Trainer(
        gpus=1 if torch.cuda.is_available() else 0,
        precision=precision,
        max_epochs=max_epochs,
        check_val_every_n_epoch=1,
        logger=HvbLogger(),
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_model_summary=False,
        enable_checkpointing=False,
        gradient_clip_val=1 if gradient_clipping else 0,
        auto_lr_find=auto_lr_finder,
        callbacks=init_callbacks(swa=stochastic_weight_average)
    )
    # eventually selects a starting learning rate
    if auto_lr_finder is True:
        trainer.tune(model,
                     train_dataloaders=dataloader_train,
                     val_dataloaders=dataloader_val)
        logging.info(f"learning rate has been set to {model.learning_rate}")
    # trains the model
    trainer.fit(model,
                train_dataloaders=dataloader_train,
                val_dataloaders=dataloader_val)
    assert not trainer.logger.logs.empty
    assert initial_weights != model.state_dict().__str__(), \
        f"model not updating"
    logs: pd.DataFrame = deepcopy(trainer.logger.logs)
    # frees some memory
    del trainer, \
        dataloader_train, dataloader_val
    return logs
