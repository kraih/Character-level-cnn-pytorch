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
from pathlib import Path
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
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    parser.add_argument("--dumps", type=str, required=True, help="Where cavil dumps are stored")
    parser.add_argument("-v", "--verbose", action='store_true', help="Preview mismatched files")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
      model = torch.load(opt.input)
    else:
      model = torch.load(opt.input, map_location='cpu')

    if torch.cuda.is_available():
        model.cuda()

    for fn in sorted(glob.glob(opt.dumps + '/bad/*.txt') + glob.glob(opt.dumps  + '/good/*.txt')):
      test_set = MyDataset(fn, opt.max_length)
      test_generator = DataLoader(test_set)

      model.eval()
      verbose = opt.verbose;
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
            if weighti != '0000' and weighti != '1000' and weighti != '0999' and weighti != '0001': 
               print(weighti, fn)
            if (weight == 1 and fn.find('/bad/') > 0) or (weight == 0 and fn.find('/good/') > 0):
               print(True if weight == 1 else False, weighti, fn)
               if verbose:
                 content = Path(fn).read_bytes()
                 try:
                    content = content.decode('utf-8')
                 except UnicodeDecodeError:
                    content = content.decode('iso-8859-15')
                 print("\n\n" + content + "\n\n")

if __name__ == "__main__":
    opt = get_args()
    train(opt)
