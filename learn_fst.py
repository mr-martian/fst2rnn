#!/usr/bin/env python3

import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle
import random

def one_hot(n, dim):
    ret = np.zeros(dim)
    ret[n] = 1
    return ret

def dtanh(x):
    return 1 - (np.tanh(x)**2)

def softmax(z):
    exps = np.exp(z - np.max(z))
    return exps / np.sum(exps)

class Model:
    def __init__(self):
        self.hidden_dim = 0
        self.embed_dim = 0
        self.in_alpha = ['@0@']
        self.out_alpha = ['@END@']
        self.embed_weights = None
        self.embed_bias = None
        self.main_weights = None
        self.main_bias = None
        self.decode_weights = None
        self.decode_bias = None
        self.epoch_losses = []
    def make_weights(self):
        self.embed_weights = np.random.rand(len(self.in_alpha) + self.hidden_dim, self.embed_dim)
        self.embed_bias = np.random.rand(self.embed_dim)
        self.main_weights = np.random.rand(self.embed_dim, self.hidden_dim*2)
        self.main_bias = np.random.rand(self.hidden_dim*2)
        self.decode_weights = np.random.rand(self.hidden_dim, len(self.out_alpha))
        self.decode_bias = np.random.rand(len(self.out_alpha))
    def predict_single(self, n, hidden):
        '''does not run softmax - returns the raw scores'''
        conc = np.concatenate((hidden, one_hot(n, len(self.in_alpha))))
        emb1 = conc @ self.embed_weights
        emb2 = np.tanh(emb1 + self.embed_bias)
        main2 = emb2 @ self.main_weights
        main3 = np.tanh(main2 + self.main_bias)
        out_hidden = main3[:self.hidden_dim]
        dec1 = main3[self.hidden_dim:] @ self.decode_weights
        dec2 = dec1 + self.decode_bias
        return dec2, out_hidden
    def predict(self, syms):
        hidden = np.zeros(self.hidden_dim)
        ret = []
        for s in syms:
            c = 0
            if s in self.in_alpha:
                c = self.in_alpha.index(s)
            out, hidden = self.predict_single(c, hidden)
            # TODO: constrain this to not produce @END@ until we
            # reach the end of the input
            ret.append((self.out_alpha[np.argmax(out)], hidden))
        # we could handle some input epsilons by at this point
        # continuing until the output is @END@
        return ret
    def backprop_single(self, x_sym, x_state, y_sym, y_state):
        x = one_hot(self.in_alpha.index(x_sym), len(self.in_alpha))
        y = one_hot(self.out_alpha.index(y_sym), len(self.out_alpha))

        emb1 = np.concatenate((x_state, x))
        emb2 = emb1 @ self.embed_weights
        emb3 = emb2 + self.embed_bias
        emb4 = np.tanh(emb3)
        main2 = emb4 @ self.main_weights
        main3 = main2 + self.main_bias
        main4 = np.tanh(main3)
        out_state = main4[:self.hidden_dim]
        dec1 = main4[self.hidden_dim:]
        dec2 = dec1 @ self.decode_weights
        dec3 = dec2 + self.decode_bias
        dec4 = softmax(dec3)

        self.losses.append(-np.sum(np.multiply(y, np.log(dec4))) + np.sum(np.power(out_state - y_state, 2)))

        rev = dec4 - y
        self.dbg.append(rev * self.decode_bias)
        rev = self.decode_weights * rev
        self.dwg.append(rev)
        rev = np.sum(rev, axis=1)
        #state_diff = out_state - y_state
        state_diff = -2 * np.multiply(y_state - out_state, out_state)
        if y_sym == '@END@':
            state_diff = np.zeros(state_diff.shape)
        rev = np.concatenate((state_diff, rev))
        rev *= dtanh(main3)
        self.mbg.append(rev * self.main_bias)
        rev = self.main_weights * rev
        self.mwg.append(rev)
        rev = np.sum(rev, axis=1)
        #rev = rev[self.hidden_dim:]
        #rev *= dtanh(emb2)
        rev *= dtanh(emb3)
        self.ebg.append(rev * self.embed_bias)
        rev = self.embed_weights * rev
        self.ewg.append(rev)
    def sample_states(self, X, Y, n=30):
        for i in range(len(X)):
            ls = [j for j in range(len(X)) if all(a == b for a,b in zip(X[i][1], Y[j][1]))]
            random.shuffle(ls)
            for idx in ls[:n]:
                _, hidden = self.predict_single(self.in_alpha.index(X[idx][0]), X[idx][1])
                self.backprop_single(X[i][0], hidden, *Y[i])
    def epoch(self, X, Y, alpha=0.01):
        self.losses = []
        self.ewg = []
        self.ebg = []
        self.mwg = []
        self.mbg = []
        self.dwg = []
        self.dbg = []
        for x, y in zip(X, Y):
            self.backprop_single(*x, *y)
        self.sample_states(X, Y)
        self.epoch_losses.append(sum(self.losses) / len(self.losses))
        print('  Loss:', self.epoch_losses[-1])
        coef = alpha / len(X)
        self.embed_weights -= coef * sum(self.ewg)
        self.embed_bias -= coef * sum(self.ebg)
        self.main_weights -= coef * sum(self.mwg)
        self.main_bias -= coef * sum(self.mbg)
        self.decode_weights -= coef * sum(self.dwg)
        self.decode_bias -= coef * sum(self.dbg)
    def save(self, fname):
        with open(fname, 'wb') as fout:
            pickle.dump([
                self.hidden_dim,
                self.embed_dim,
                self.in_alpha,
                self.out_alpha,
                self.embed_weights,
                self.embed_bias,
                self.main_weights,
                self.main_bias,
                self.decode_weights,
                self.decode_bias,
            ], fout)
    def load(fname):
        with open(fname, 'rb') as fin:
            m = Model()
            blob = pickle.load(fin)
            m.hidden_dim = blob[0]
            m.embed_dim = blob[1]
            m.in_alpha = blob[2]
            m.out_alpha = blob[3]
            m.embed_weights = blob[4]
            m.embed_bias = blob[5]
            m.main_weights = blob[6]
            m.main_bias = blob[7]
            m.decode_weights = blob[8]
            m.decode_bias = blob[9]
            return m

class ATT:
    def __init__(self):
        self.states = set()
        self.in_symbols = set(['@0@'])
        self.out_symbols = set(['@0@', '@END@'])
        self.transitions = [] # (src, dest, in, out)
    def read(fin):
        ret = ATT()
        for line in fin:
            ls = line.strip().split('\t')
            if not ls:
                continue
            if len(ls) >= 4:
                src, dest, sin, sout = ls[:4]
                src = int(src)
                dest = int(dest)
                ret.states.add(src)
                ret.states.add(dest)
                ret.in_symbols.add(sin)
                ret.out_symbols.add(sout)
                ret.transitions.append((src, dest, sin, sout))
            else:
                st = int(ls[0])
                ret.states.add(st)
                ret.transitions.append((st, -1, '@0@', '@END@'))
        return ret
    def to_model(self):
        m = Model()
        m.in_alpha = sorted(self.in_symbols)
        m.out_alpha = sorted(self.out_symbols)
        m.hidden_dim = int(np.ceil(np.log2(len(self.states))))
        #m.hidden_dim = len(self.states)
        X = []
        Y = []
        # TODO: we can at minimum be clever and merge states that
        # never distinguish anything
        def hide(n):
            ret = np.zeros(m.hidden_dim)
            if n < 0:
                return ret
            for i, c in enumerate(reversed(bin(n)[2:])):
                if c == '1':
                    ret[i] = 1
            return ret
        for src, dest, sin, sout in self.transitions:
            X.append((sin, hide(src)))
            Y.append((sout, hide(dest)))
            #X.append((sin, one_hot(src, m.hidden_dim)))
            #Y.append((sout, one_hot(dest, m.hidden_dim)))
        return m, X, Y

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('convert an FST into an RNN')
    parser.add_argument('--model', action='store')
    parser.add_argument('--att', action='store')
    parser.add_argument('--input', action='store')
    parser.add_argument('--seed', type=int, default=2319)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--alpha', type=float, default=0.01)
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()

    np.random.seed(args.seed)
    model = None
    if args.att:
        with open(args.att) as fin:
            a = ATT.read(fin)
        model, train_X, train_Y = a.to_model()
        model.embed_dim = 2
        model.make_weights()
        for i in range(args.epochs):
            print(f'starting epoch {i+1}')
            model.epoch(train_X, train_Y, args.alpha)
            #if i > 1 and model.epoch_losses[-1] > model.epoch_losses[-2]:
            #    print('  early stopping')
            #    break
        if args.graph:
            plt.plot(list(range(len(model.epoch_losses))), model.epoch_losses)
            plt.show()
        if args.model:
            model.save(args.model)
    elif args.model:
        model = Model.load(args.model)
        if args.input:
            print(model.predict(args.input.split()))
        else:
            print('No input provided, nothing to do')
    else:
        print('No ATT or model file provided, nothing to do')
