import argparse
import random
from os import makedirs
from os.path import isdir
from typing import Dict, Union


def get_args() -> Dict[str, Union[bool, str, int, float]]:
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        help="The seed for reproducibility")
    parser.add_argument("--max_sentences_length",
                        type=int,
                        default=64,
                        help="The maximum length to which the training sentences are truncated or padded")

    # training args
    parser.add_argument("--batch_size",
                        default=8,
                        type=int,
                        help="Type of validation algorithm ('kfold' or 'loso')")
    parser.add_argument("--min_epochs",
                        default=1,
                        type=int,
                        help="Minimum number of epochs")
    parser.add_argument("--max_epochs",
                        default=1000,
                        type=int,
                        help="Maximum number of epochs")
    parser.add_argument("--precision",
                        default=32,
                        type=int,
                        choices={16, 32},
                        help="Whether to use 32- ore 16-bit precision")
    parser.add_argument("--train_set_size",
                        default=0.8,
                        type=float,
                        help="The size of the training set, in percentage")

    # model args
    parser.add_argument("--num_encoders",
                        default=2,
                        type=int,
                        help="Number of encoders")
    parser.add_argument("--num_decoders",
                        default=2,
                        type=int,
                        help="Number of decoders")
    parser.add_argument("--embeddings_dim",
                        default=128,
                        type=int,
                        help="Dimension of the internal embedding")
    parser.add_argument("--auto_lr_finder",
                        default=False,
                        action="store_true",
                        help="Whether to run an automatic learning range finder algorithm")

    # regularization
    parser.add_argument("--dropout_p",
                        default=0.2,
                        type=float,
                        help="The amount of dropout to use")
    parser.add_argument("--noise_strength",
                        default=0,
                        type=float,
                        help="The amount of gaussian noise to add to the eegs")
    parser.add_argument("--gradient_clipping",
                        default=False,
                        action="store_true",
                        help="Whether to clip the gradients to 1")
    parser.add_argument("--stochastic_weight_average",
                        default=False,
                        action="store_true",
                        help="Whether to use the SWA algorithm")
    parser.add_argument("--disable_masking",
                        default=False,
                        action="store_true",
                        help="Whether not to mask a percentage of embeddings during training of FEEGT")
    parser.add_argument("--learning_rate",
                        default=1e-3,
                        type=float,
                        help="Learning rate of the model")

    args = parser.parse_args()

    assert 0 < args.train_set_size < 1
    args.val_set_size = 1 - args.train_set_size
    assert args.train_set_size + args.val_set_size == 1

    assert args.batch_size >= 1
    assert args.min_epochs >= 1
    assert args.max_epochs >= 1 and args.max_epochs >= args.min_epochs
    if args.seed is None:
        args.seed = random.randint(0, 1000000)

    assert args.num_encoders >= 1
    assert args.num_decoders >= 1
    assert 0 <= args.dropout_p < 1
    assert args.noise_strength >= 0
    assert args.learning_rate > 0

    return vars(args)
