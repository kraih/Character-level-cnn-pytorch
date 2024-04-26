# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import glob
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.utils import *
from src.dataset import MyDataset
from src.character_level_cnn import CharacterLevelCNN


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of the model described in the paper: Character-level convolutional networks for text classification"""
    )
    parser.add_argument(
        "-a",
        "--alphabet",
        type=str,
        default="""abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""",
    )
    parser.add_argument("-m", "--max_length", type=int, default=1014)
    parser.add_argument(
        "-i", "--input", type=str, default="input", help="path to input folder"
    )
    parser.add_argument(
        "--dumps", type=str, required=True, help="Where cavil dumps are stored"
    )
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        model = torch.load(opt.input)
    else:
        model = torch.load(opt.input, map_location="cpu")

    if torch.cuda.is_available():
        model.cuda()

    files = sorted(
        glob.glob(opt.dumps + "/bad/*.txt") + glob.glob(opt.dumps + "/good/*.txt")
    )
    correct = 0
    for fn in files:
        is_legal_text = os.path.basename(os.path.dirname(fn)) == "good"
        test_set = MyDataset(fn, opt.max_length)
        test_generator = DataLoader(test_set)

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
                response = {"license": (True if weight == 1 else False)}
                response["filename"] = fn
                if weight == 1:
                    response["confidence"] = (weighti - 500) / 5
                else:
                    response["confidence"] = (500 - weighti) / 5
                print(f"{is_legal_text}: {str(response)}")

                if is_legal_text and response["license"] == True:
                    correct += 1
                elif not is_legal_text and response["license"] == False:
                    correct += 1

    print(f"Accuracy: {correct / len(files)}")


if __name__ == "__main__":
    opt = get_args()
    train(opt)
