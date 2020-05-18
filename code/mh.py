# -*- coding: utf-8 -*-
"""
Created on Mon May 18 15:26:07 2020

@author: 63561
"""

import torch
import numpy
import copy

import util


class MetropolisHastings(object):
    
    def __init__(self, vocab, forward_lm, backward_lm=None, device=None):
        
        self.__vocab = vocab
        self.__forward_lm = forward_lm
        self.__backward_lm = backward_lm
        if device is None:
            self.__device = torch.device("cpu")
        else:
            self.__device = device

    def replace(self, ids, idx, n_cand=50):
        
        forward_X = torch.tensor([[self.__vocab.word2idx["<eos>"]]+ids+[self.__vocab.word2idx["<eos>"]]])
        forward_X = forward_X.type(torch.int64).to(self.__device)
        forward_idx = idx
        if self.__backward_lm is not None:
            backward_X = util.reverse_batch(forward_X, self.__device)
            backward_idx = len(ids) - idx - 1
            backward_X = backward_X[:, :-1]
            with torch.no_grad():
                backward_log_prob = self.__backward_lm(backward_X)
        forward_X = forward_X[:, :-1]
        with torch.no_grad():
            forward_log_prob = self.__forward_lm(forward_X)
        log_prob = forward_log_prob[0, forward_idx, :]
        if self.__backward_lm is not None:
            log_prob += backward_log_prob[0, backward_idx, :]
        cand_ids = torch.argsort(log_prob)[-n_cand:]
        if int(ids[idx]) not in cand_ids:
            cand_ids[0] = int(ids[idx])
            ori_idx = 0
        else:
            ori_idx = list(cand_ids.cpu().numpy()).index(int(ids[idx]))
        
        forward_X = []
        for c in cand_ids:
            new_ids = copy.deepcopy(ids)
            new_ids[idx] = c
            forward_X.append([self.__vocab.word2idx["<eos>"]]+new_ids+[self.__vocab.word2idx["<eos>"]])
        forward_X = torch.tensor(forward_X).type(torch.int64).to(self.__device)
        if self.__backward_lm is not None:
            backward_X = util.reverse_batch(forward_X, self.__device)
            backward_X, backward_Y = backward_X[:, :-1], backward_X[:, 1:]
            with torch.no_grad():
                backward_log_prob = self.__backward_lm(backward_X)
            backward_log_prob = util.log_seq_prob(backward_log_prob, backward_Y, self.__device)
        forward_X, forward_Y = forward_X[:, :-1], forward_X[:, 1:]
        with torch.no_grad():
            forward_log_prob = self.__forward_lm(forward_X)
        log_prob = util.log_seq_prob(forward_log_prob, forward_Y, self.__device)
        if self.__backward_lm is not None:
            log_prob += backward_log_prob
        with torch.no_grad():
            prob = torch.pow(numpy.e, log_prob - log_prob.min())
        cand_sel = util.random_sample_from_unnormalized_prob(prob)
        prob = prob.cpu().numpy()
        
        return cand_ids[cand_sel], prob[ori_idx], prob[cand_sel]
    
if __name__ == "__main__":
    
    import argparse
    import random
    import os
    
    import data
    import model
    
    parser = argparse.ArgumentParser(description='PyTorch Poetry Language Model')
    parser.add_argument('--seed', type=int, default=1726,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--gpu', type=str, default="0",
                        help='the selected gpu')
 
    args = parser.parse_args()
    
    emsize = 256
    nhid = 512
    nlayers = 2
    nhead = 8
    dropout = 0.2
    forward_save = '../model/transformer_embed_256_hidden_512_layer_2_head_8.pt'
    backward_save = '../model/transformer_embed_256_hidden_512_layer_2_head_8_rev.pt'
    
    # Set the random seed manually for reproducibility.
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
    device = torch.device("cuda" if args.cuda else "cpu")
    
    ###############################################################################
    # Load vocab
    ###############################################################################
    
    vocab = data.Dictionary()
    with open("../data/train.txt", 'r', encoding="utf8") as f:
        for line in f:
            words = line.split() + ['<eos>']
            for word in words:
                vocab.add_word(word)
    
    ###############################################################################
    # Build the model
    ###############################################################################
    
    ntokens = len(vocab)
    forward_model = model.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    backward_model = model.TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)
    
    # Load the best saved model.
    with open(forward_save, 'rb') as f:
        forward_model = torch.load(f)
    with open(backward_save, 'rb') as f:
        backward_model = torch.load(f)
        
    mh = MetropolisHastings(vocab, forward_model, backward_model, device)
             
    ###############################################################################
    # Run the model
    ###############################################################################
    
    max_len = 16
    n_rand = 5
    
    n_cand = 100
    n_iter = max_len * 1
    relax = 0.99
    
    while True:
        print("*** GENERATION PHASE ***")
        _input = input("Begin with:\t")
        if len(_input) <= 0:
            _input = []
        _input = [i for i in _input]
        ids = []
        for w in _input:
            ids.append(vocab.get_idx(w))
        for i in ids:
            print (vocab.idx2word[i], end="")
        print ("")
        poetry = torch.tensor([[vocab.word2idx["<eos>"]]+ids]).type(torch.int64).to(device)
        poetry = torch.transpose(poetry, 1, 0)
        
        while len(ids) < max_len:
            with torch.no_grad():
                output = forward_model(poetry)
                output = output.view(-1, ntokens)
                nxt_w = random.sample(list(torch.argsort(output[-1])[-n_rand:].cpu().numpy()), 1)
                #print (list(torch.argsort(output[-1])[-n_rand:].cpu().numpy()), nxt_w)
                nxt_w = nxt_w[0]
                if nxt_w == vocab.word2idx["<eos>"]:
                    break
                ids.append(nxt_w)
            for i in ids:
                print (vocab.idx2word[i], end="")
            print ("")
            poetry = torch.tensor([[vocab.word2idx["<eos>"]]+ids]).type(torch.int64).to(device)
            poetry = torch.transpose(poetry, 1, 0)
    
        print("*** REFINEMENT PHASE ***")
        _input = [vocab.idx2word[i] for i in ids]
        print ("Original Poetry", end=":\t")
        for i in _input:
            print (i, end="")
        print ()
        idx, it = 0, 0
        shutdown_cnt = 0
        while it < n_iter:
            if _input[idx] in ["，", "。", "？", "！", "；", "：", ",", ".", "?", "!", ":", ";"]:
                idx = (idx + 1) % len(_input)
                shutdown_cnt += 1
                continue
            original_ch = _input[idx]
            _id, old_prob, new_prob = mh.replace(ids, idx, n_cand)
            if new_prob < old_prob * relax or vocab.idx2word[_id] in [original_ch, "<unk>", "<eos>"] \
                or vocab.idx2word[_id] in ["，", "。", "？", "！", "；", "：", ",", ".", "?", "!", ":", ";"]:
                #print ("Iter %d %s => %s (%.1e) :\tREJECT!" \
                #       % (it+1, original_ch, vocab.idx2word[_id], new_prob/old_prob))
                idx = (idx + 1) % len(_input)
                shutdown_cnt += 1
                if shutdown_cnt >= len(_input):
                    break
                continue
            shutdown_cnt = 0
            ids[idx] = _id
            _input[idx] = vocab.idx2word[_id]
            print ("Iter %d %s => %s (%.1e) :" \
                   % (it+1, original_ch, vocab.idx2word[_id], new_prob/old_prob), end="\t")
            for i in _input:
                print (i, end="")
            print()
            it += 1
            idx = (idx + 1) % len(_input)