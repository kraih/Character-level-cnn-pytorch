# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
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
    parser.add_argument("-f", "--feature", type=str, choices=["large", "small", "tiny"], default="large",
                        help="small for 256 conv feature map, large for 1024 conv feature map")
    parser.add_argument("-p", "--optimizer", type=str, choices=["sgd", "adam"], default="sgd")
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-n", "--num_epochs", type=int, default=100)
    parser.add_argument("-l", "--lr", type=float, default=0.01)  # recommended learning rate for sgd is 0.01, while for adam is 0.001
    parser.add_argument("-y", "--es_min_delta", type=float, default=0.0,
                        help="Early stopping's parameter: minimum change loss to qualify as an improvement")
    parser.add_argument("-w", "--es_patience", type=int, default=3,
                        help="Early stopping's parameter: number of epochs with no improvement after which training will be stopped. Set to 0 to disable this technique.")
    parser.add_argument("-i", "--input", type=str, default="input", help="path to input folder")
    parser.add_argument("-o", "--output", type=str, default="output", help="path to output folder")
    parser.add_argument("-v", "--log_path", type=str, default="tensorboard/char-cnn")
    parser.add_argument("--dumps", type=str, required=True, help="Where cavil dumps are stored")
    args = parser.parse_args()
    return args


def train(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(121)
    else:
        torch.manual_seed(121)
    if not os.path.exists(opt.output):
        os.makedirs(opt.output)
    output_file = open(opt.output + os.sep + "logs.txt", "w")
    output_file.write("Model's parameters: {}".format(vars(opt)))

    training_params = {"batch_size": opt.batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    test_params = {"batch_size": opt.batch_size,
                   "shuffle": False,
                   "num_workers": 0}
    training_set = MyDataset(None, None, opt.max_length, opt.dumps)
    print(len(training_set))
    test_size = int(.1 * len(training_set))
    training_set, test_set = torch.utils.data.random_split(training_set, [len(training_set) - test_size, test_size])
    print(len(training_set))
    print(len(test_set))

    training_generator = DataLoader(training_set, **training_params)
    test_generator = DataLoader(test_set, **test_params)

    if opt.feature == "small":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=2,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=256, n_fc_neurons=1024)
    elif opt.feature == "tiny":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=2,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=256, n_fc_neurons=128)
    elif opt.feature == "large":
        model = CharacterLevelCNN(input_length=opt.max_length, n_classes=2,
                                  input_dim=len(opt.alphabet),
                                  n_conv_filters=1024, n_fc_neurons=2048)
    else:
        sys.exit("Invalid feature mode!")

    log_path = "{}_{}".format(opt.log_path, opt.feature)
    if os.path.isdir(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path)
    #writer = SummaryWriter(log_path)

    if torch.cuda.is_available():
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    if opt.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=opt.lr, momentum=0.9)
    best_loss = 1e5
    best_epoch = 0
    model.train()
    num_iter_per_epoch = len(training_generator)

    for epoch in range(opt.num_epochs):
        for iter, batch in enumerate(training_generator):
            feature, label = batch
            if torch.cuda.is_available():
                feature = feature.cuda()
                label = label.cuda()
            optimizer.zero_grad()
            predictions = model(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()

            training_metrics = get_evaluation(label.cpu().numpy(), predictions.cpu().detach().numpy(),
                                              list_metrics=["accuracy"])
            print("Epoch: {}/{}, Iteration: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
                epoch + 1,
                opt.num_epochs,
                iter + 1,
                num_iter_per_epoch,
                optimizer.param_groups[0]['lr'],
                loss, training_metrics["accuracy"]))
            #writer.add_scalar('Train/Loss', loss, epoch * num_iter_per_epoch + iter)
            #writer.add_scalar('Train/Accuracy', training_metrics["accuracy"], epoch * num_iter_per_epoch + iter)
        model.eval()
        loss_ls = []
        te_label_ls = []
        te_pred_ls = []
        for batch in test_generator:
            te_feature, te_label = batch
            num_sample = len(te_label)
            if torch.cuda.is_available():
                te_feature = te_feature.cuda()
                te_label = te_label.cuda()
            with torch.inference_mode():
                te_predictions = model(te_feature)
            te_loss = criterion(te_predictions, te_label)
            loss_ls.append(te_loss * num_sample)
            te_label_ls.extend(te_label.clone().cpu())
            te_pred_ls.append(te_predictions.clone().cpu())

        te_loss = sum(loss_ls) / test_set.__len__()
        te_pred = torch.cat(te_pred_ls, 0)
        te_label = np.array(te_label_ls)
        test_metrics = get_evaluation(te_label, te_pred.numpy(), list_metrics=["accuracy", "confusion_matrix"])
        output_file.write(
            "Epoch: {}/{} \nTest loss: {} Test accuracy: {} \nTest confusion matrix: \n{}\n\n".format(
                epoch + 1, opt.num_epochs,
                te_loss,
                test_metrics["accuracy"],
                test_metrics["confusion_matrix"]))
        print("Epoch: {}/{}, Lr: {}, Loss: {}, Accuracy: {}".format(
            epoch + 1,
            opt.num_epochs,
            optimizer.param_groups[0]['lr'],
            te_loss, test_metrics["accuracy"]))
        #writer.add_scalar('Test/Loss', te_loss, epoch)
        #writer.add_scalar('Test/Accuracy', test_metrics["accuracy"], epoch)
        model.train()
        if te_loss + opt.es_min_delta < best_loss:
            best_loss = te_loss
            best_epoch = epoch
            torch.save(model, "{}/char-cnn_{}".format(opt.output, opt.feature))
            print("The model has been saved in epoch {}".format(epoch))
        # Early stopping
        if epoch - best_epoch > opt.es_patience > 0:
            print("Stop training at epoch {}. The lowest loss achieved is {} at epoch {}".format(epoch, best_loss, best_epoch))
            break
        if opt.optimizer == "sgd" and epoch % 3 == 0 and epoch > 0:
            current_lr = optimizer.state_dict()['param_groups'][0]['lr']
            current_lr /= 2
            for param_group in optimizer.param_groups:
                param_group['lr'] = current_lr


if __name__ == "__main__":
    opt = get_args()
    train(opt)
