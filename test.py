# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import shutil

from src.utils import *
from src.dataset import MyDataset
from src.character_level_cnn import CharacterLevelCNN


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Character-level convolutional networks for text classification""")
    parser.add_argument("-a", "--alphabet", type=str,
                        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small"], default="small",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    parser.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-b", "--batch_size", type=int, default=128)
    parser.add_argument("-n", "--num_epochs", type=int, default=20)
    parser.add_argument("-l", "--lr", type=float, default=0.001)  # recommended learning rate for sgd is 0.01, while for adam is 0.001
    parser.add_argument("-d", "--dataset", type=str,
                        choices=["agnews", "dbpedia", "yelp_review", "yelp_review_polarity", "amazon_review",
                                 "amazon_polarity", "sogou_news", "yahoo_answers"], default="yelp_review_polarity",
                        help="public dataset used for experiment. If this parameter is set, parameters input and output are ignored")
    parser.add_argument("-y", "--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("-w", "--es_patience", type=int, default=3,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    parser.add_argument("-v", "--log_path", type=str, default="tensorboard/char-cnn")
    args = parser.parse_args()
    return args


def train(opt):
    model = torch.load(opt.input)

    if torch.cuda.is_available():
        model.cuda()

    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    for fn in glob.glob('/space/SP/bad/*.txt'):
      test_set = MyDataset(fn, opt.max_length)
      test_generator = DataLoader(test_set, **test_params)

      model.eval()
      for batch in test_generator:
        te_feature, te_label = batch
        if torch.cuda.is_available():
            te_feature = te_feature.cuda()
            te_label = te_label.cuda()
        with torch.no_grad():
            te_predictions = model(te_feature)
            out = F.softmax(te_predictions, 1)
            weight = torch.argmax(out[0])
            weighti = int(out[0][1].item() * 1000)
            weighti = '%04d' % weighti
            print(True if weight == 1 else False, weighti, fn)

if __name__ == "__main__":
    opt = get_args()
    train(opt)
