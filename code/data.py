# -*- coding: utf-8 -*-
"""
Created on Sun May 17 19:47:05 2020

@author: 63561
"""

import os
from io import open
import torch

class Dictionary(object):
    def __init__(self):
        self.word2idx = {"<unk>": 0, "<eos>": 1}
        self.idx2word = ["<unk>", "<eos>"]

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)
    
    def get_idx(self, word):
        if word in self.word2idx.keys():
            return self.word2idx[word]
        else:
            return self.word2idx["<unk>"]


class Corpus(object):
    def __init__(self, path="../data"):
        self.dictionary = Dictionary()
        self.add_dict(os.path.join(path, 'train.txt'))
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'dev.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))
    
    def add_dict(self, path):
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r', encoding="utf8") as f:
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    self.dictionary.add_word(word)

    def tokenize(self, path):
        """Tokenizes a text file."""
        # Tokenize file content
        with open(path, 'r', encoding="utf8") as f:
            idss = []
            for line in f:
                words = line.split() + ['<eos>']
                ids = []
                for word in words:
                    ids.append(self.dictionary.get_idx(word))
                idss.append(torch.tensor(ids).type(torch.int64))
            ids = torch.cat(idss)
        return ids
    
if __name__ == "__main__":
    
    poetry = Corpus()