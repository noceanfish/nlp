import numpy as np
import pandas as pd
import os
import sys
import string
import operator


def init_weight_and_bias(M1, M2):
    W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
    b = np.zeros(M2)
    return W.astype(np.float32), b.astype(np.float32)


def init_filter(shape, poolsz):
    w = np.random.randn(*shape) / np.sqrt(np.prod(shape[1:]) + shape[0]*np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)


def relu(x):
    return x * (x > 0)


def sigmoid(A):
    return 1 / (1 + np.exp(-A))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)


def sigmoid_cost(T, Y):
    return -(T*np.log(Y) + (1-T)*np.log(1-Y)).sum()


def cost(T, Y):
    return -(T * np.log(Y)).sum()


def cost2(T, Y):
    N = len(T)
    t = -np.log(Y[np.arange(N), T]).sum()
    return -np.log(Y[np.arange(N), T]).sum()


def error_rate(targets, predictions):
    return np.mean(targets != predictions)


def y2indicator(y):
    N = len(y)
    K = len(set(y))
    ind = np.zeros((N, K))
    for i in range(N):
        ind[i, y[i]] = 1
    return ind


def getData(balance_ones=True):
    X = []
    Y = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            X.append([int(p) for p in row[1].split()])
            Y.append(int(row[0]))

    X, Y = np.array(X) / 255.0, np.array(Y)

    if balance_ones:
        X0, Y0 = X[Y!=1, :], Y[Y!=1]
        X1 = X[Y==1, :]
        X1 = np.repeat(X1, 9, axis=0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1]*len(X1)))

    return X, Y


def getImageData():
    X, Y = getData()
    N, D = X.shape
    d = int(np.sqrt(D))
    x = X.reshape(N, 1, d, d)
    return X, Y


def getBinaryData():
    X = []
    Y = []
    first = True
    for line in open('fer2013.csv'):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)


def find_analogies(w1, w2, w3, We, word2idx, idx2word):
    V, D = We.shape

    king = We[word2idx[w1]]
    man = We[word2idx[w2]]
    woman = We[word2idx[w3]]
    v0 = king - man + woman

    for dist in ('euclidean', 'cosine'):
        distances = pairwise_distances(v0.reshape(1, D), We, metric=dist).reshape(V)
        # idx = distances.argmin()
        # best_word = idx2word[idx]
        idx = distances.argsort()[:4]
        best_idx = -1
        keep_out = [word2idx[w] for w in (w1, w2, w3)]
        for i in idx:
            if i not in keep_out:
                best_idx = i
                break
        best_word = idx2word[best_idx]


        print("closest match by", dist, "distance:", best_word)
        print(w1, "-", w2, "=", best_word, "-", w3)


# unfortunately Python 2 and 3 translates work differently
def remove_punctuation_2(s):
    return s.translate(None, string.punctuation)


def remove_punctuation_3(s):
    return s.translate(str.maketrans('','',string.punctuation))


if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3


def my_tokenizer(s):
    s = remove_punctuation(s)
    s = s.lower()  # downcase
    return s.split()


def get_wikipedia_data(n_files, n_vocab, by_paragraph=False):
    prefix = './large_files/'

    if not os.path.exists(prefix):
        print("Are you sure you've downloaded, converted, and placed the Wikipedia data into the proper folder?")
        print("I'm looking for a folder called large_files, adjacent to the class folder, but it does not exist.")
        print("Please download the data from https://dumps.wikimedia.org/")
        print("Quitting...")
        exit()

    input_files = [f for f in os.listdir(prefix) if f.startswith('enwiki') and f.endswith('txt')]

    if len(input_files) == 0:
        print("Looks like you don't have any data files, or they're in the wrong location.")
        print("Please download the data from https://dumps.wikimedia.org/")
        print("Quitting...")
        exit()

    # return variables
    sentences = []
    word2idx = {'START': 0, 'END': 1}
    idx2word = ['START', 'END']
    current_idx = 2
    word_idx_count = {0: float('inf'), 1: float('inf')}

    if n_files is not None:
        input_files = input_files[:n_files]

    for f in input_files:
        print("reading:", f)
        for line in open(prefix + f, encoding='utf-8'):
            line = line.strip()
            # don't count headers, structured data, lists, etc...
            if line and line[0] not in ('[', '*', '-', '|', '=', '{', '}'):
                if by_paragraph:
                    sentence_lines = [line]
                else:
                    sentence_lines = line.split('. ')
                for sentence in sentence_lines:
                    tokens = my_tokenizer(sentence)
                    for t in tokens:
                        if t not in word2idx:
                            word2idx[t] = current_idx
                            idx2word.append(t)
                            current_idx += 1
                        idx = word2idx[t]
                        word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
                    sentence_by_idx = [word2idx[t] for t in tokens]
                    sentences.append(sentence_by_idx)

    # restrict vocab size
    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small['UNKNOWN'] = new_idx
    unknown = new_idx

    assert('START' in word2idx_small)
    assert('END' in word2idx_small)
    assert('king' in word2idx_small)
    assert('queen' in word2idx_small)
    assert('man' in word2idx_small)
    assert('woman' in word2idx_small)

    # map old idx to new idx
    sentences_small = []
    for sentence in sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small