import argparse
# import os
# from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialize_parser()

    def initialize_parser(self):
        self.parser.add_argument('--target_model', type=str, default="huggyllama/llama-7b", help="the model to attack: huggyllama/llama-65b, text-davinci-003")
        self.parser.add_argument('--output_dir', type=str, default="/dt/shabtaia/dt-sicpa/eyal/Tabular")
        self.parser.add_argument('--data', type=str, default="WTQ", help="the dataset to evaluate: default is WikiMIA")
        self.parser.add_argument('--split', type=str, default="train", help="the split of the dataset to evaluate: default is train")
        self.parser.add_argument('--top_k', type=int, default=-1, help="the number of samples to evaluate")
        self.parser.add_argument('--seed', type=int, default=42, help="random seed")
        self.parser.add_argument('--use_existing', type=str, default="all", help="use existing data, model or all")
        self.parser.add_argument('--num_epochs', type=int, default=1, help="the number of epochs to train the model")
        self.parser.add_argument('--table_encoding', type=str, default="line-sep", help="the table encoder to use")
        self.parser.add_argument('--max_table_size', type=int, default=-1, help="the maximum size of the table")

        self.parser.add_argument('--syn_data', type=str, default="synthetic_data.csv", help="the synthetic data to use")