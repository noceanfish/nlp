import json
import numpy as np
import matplotlib as plt
from scipy.special import expit as sigmoid
from sklearn.utils import shuffle
from datetime import datetime

from scipy.spatial.distance import cosine as cos_dist
from sklearn.metrics.pairwise import pairwise_distances

from glob import glob

import os
import sys
import string

sys.path.append(os.path.abspath('..'))
# from rnn_class.brown import get_sentences_with_word2vec_limit_vocab as get_brown


def remove_punctuation_2(s):
    return s.traslate(None, string.punctuation)


def remove_punctuation_3(s):
    return s.traslate(str.maketrans('', '', string.punctuation))


if sys.version.startswith('2'):
    remove_punctuation = remove_punctuation_2
else:
    remove_punctuation = remove_punctuation_3


def get_wiki():
    V = 20000
    file = glob('../large_files/enwiki*.txt')
    all_word_counts = {}
    for f in file:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        if word not in all_word_counts:
                            all_word_counts[word] = 0
                        all_word_counts[word] += 1
    print("finish counting!")

    V = min(V, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V-1]] + ['<UNK>']
    word2idx = {w:i for i, w in enumerate(top_words)}
    unk = word2idx['<UNK>']

    sents = []
    for f in file:
        for line in open(f):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    sent = [word2idx[w] if w in word2idx else unk for w in s]
                    # sent = []
                    # for w in s:
                    #     if w in word2idx:
                    #         sent.append(word2idx[w])
                    #     else:
                    #         sent.append(unk)
                    sents.append(sent)
    return sents, word2idx


def train_model(savedir):
    sentences, word2idx = get_wiki()
    vocab_size = len(word2idx)

    # config
    window_size = 5
    learning_rate = 0.025
    final_learning_rate = 0.0001
    num_negaticves = 5
    epochs = 20
    D =50

    learning_rate_delta = (learning_rate - final_learning_rate) / epochs

    W = np.random.randn(vocab_size, D)
    V = np.random.randn(D, vocab_size)

    p_neg = get_negatve_sampling_distribution(sentences, vocab_size)
    costs = []

    total_words = sum(len(sentences), for sentence in sentences)
    print("total word in corpus:", total_words)

    threshold = 1e-5
    p_drop = 1 - np.sqrt(threshold / p_neg)

    for epoch in range(epochs):
        np.random.shuffle(sentences)

        cost = 0
        counter = 0
        t0 = datetime.now()
        for sentence in sentences:
            sentence = [w for w in sentence if np.random.random() < 1-p_drop[w]]
            if len(sentence) < 2:
                continue

            randomly_ordered_positions = np.random.choice(len(sentence), size=len(sentence), replace=False)

            for pos in randomly_ordered_positions:
                word = sentence[pos]

                context_words = get_context(pos, sentence, window_size)
                neg_word = np.random.choice(vocab_size, p-p_neg)
                targets = np.array(context_words)

                c = sgd(word, targets, 1, learning_rate, W, V)
                cost += c
                c = sgd(neg_word, targets, 0, learning_rate, W, V)
                cost += c

            counter += 1
            if counter % 100 == 0:
                sys.stdout.write("processed %s / %s\r" % (counter, len(sentences)))
                sys.stdout.flush()

