# -*- coding: utf-8 -*-
"""
Created on Mon May 18 11:07:35 2020

@author: 63561
"""

import argparse
import random
import os
import torch

import data
import model

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch Poetry Language Model')
    parser.add_argument('--model', type=str, default='Transformer',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU, Transformer)')
    parser.add_argument('--emsize', type=int, default=256,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=512,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.5,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=50,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=200,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1726,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=50, metavar='N',
                        help='report interval')
    parser.add_argument('--save', type=str, default='../model/transformer_embed_256_hidden_512_layer_2_head_8.pt',
                        help='path to save the final model')
    parser.add_argument('--onnx-export', type=str, default='',
                        help='path to export the final model in onnx format')
    
    parser.add_argument('--gpu', type=str, default="0",
                        help='the selected gpu')
    parser.add_argument('--nhead', type=int, default=8,
                        help='the number of heads in the encoder/decoder of the transformer model')
    
    args = parser.parse_args()
    
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
    if args.model == 'Transformer':
        model = model.TransformerModel(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout).to(device)
    else:
        model = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)
    
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f)
        # after load the rnn params are not a continuous chunk of memory
        # this makes them a continuous chunk, and will speed up forward pass
        # Currently, only rnn model supports flatten_parameters function.
        if args.model in ['RNN_TANH', 'RNN_RELU', 'LSTM', 'GRU']:
            model.rnn.flatten_parameters()
             
    ###############################################################################
    # Run the model
    ###############################################################################
    
    while True:
        print()
        _input = input("诗句的开头：")
        if len(_input) <= 0:
            _input = []
        n_rand = 3
        _input = [i for i in _input]
        ids = []
        for w in _input:
            ids.append(vocab.get_idx(w))
        for i in ids:
            print (vocab.idx2word[i], end="")
        print ("", end="\r")
        poetry = torch.tensor([[vocab.word2idx["<eos>"]]+ids]).type(torch.int64).to(device)
        poetry = torch.transpose(poetry, 1, 0)
        
        while len(ids) < args.bptt:
            with torch.no_grad():
                if args.model == 'Transformer':
                    output = model(poetry)
                    output = output.view(-1, ntokens)
                else:
                    output, _ = model(poetry, model.init_hidden(1))
                nxt_w = random.sample(list(torch.argsort(output[-1])[-n_rand:].cpu().numpy()), 1)
                #print (list(torch.argsort(output[-1])[-n_rand:].cpu().numpy()), nxt_w)
                nxt_w = nxt_w[0]
                if nxt_w == vocab.word2idx["<eos>"]:
                    break
                ids.append(nxt_w)
            for i in ids:
                print (vocab.idx2word[i], end="")
            print ("", end="\r")
            poetry = torch.tensor([[vocab.word2idx["<eos>"]]+ids]).type(torch.int64).to(device)
            poetry = torch.transpose(poetry, 1, 0)