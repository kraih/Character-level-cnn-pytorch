# -*- coding: utf-8 -*-
"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import numpy as np
import glob
import sys
import csv
from torch.utils.data import Dataset
csv.field_size_limit(sys.maxsize)


def load_dump(fn):
    f = open(fn)
    try:
        return f.read()
    except UnicodeDecodeError:
        pass
    f.close()
    f = open(fn, encoding='iso-8859-15')
    return f.read()


class MyDataset(Dataset):
    def __init__(self, data_path, data=None, max_length=1014, dumps_folder="/space/SP"):
        self.data_path = data_path
        self.vocabulary = list(
            """abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}""")
        self.identity_mat = np.identity(len(self.vocabulary))
        texts, labels = [], []

        if not data_path:
            if data:
                try:
                    data = data.decode('utf-8')
                except UnicodeDecodeError:
                    data = data.decode('iso-8859-15')
                
                texts.append(data)
                labels.append(0)
            else:
                for fn in glob.glob(dumps_folder + '/good/*.txt'):
                    texts.append(load_dump(fn))
                    labels.append(1)
                for fn in glob.glob(dumps_folder + '/bad/*.txt'):
                    texts.append(load_dump(fn))
                    labels.append(0)
        else:
            texts.append(load_dump(data_path))
            labels.append(1)

        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        self.length = len(self.labels)
        self.num_classes = len(set(self.labels))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raw_text = self.texts[index]
        data = np.array([self.identity_mat[self.vocabulary.index(i)] for i in list(raw_text) if i in self.vocabulary],
                        dtype=np.float32)
        if len(data) > self.max_length:
            data = data[:self.max_length]
        elif 0 < len(data) < self.max_length:
            data = np.concatenate(
                (data, np.zeros((self.max_length - len(data), len(self.vocabulary)), dtype=np.float32)))
        elif len(data) == 0:
            data = np.zeros((self.max_length, len(self.vocabulary)), dtype=np.float32)
        label = self.labels[index]
        return data, label
