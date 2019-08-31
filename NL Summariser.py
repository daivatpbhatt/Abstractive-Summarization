#!/usr/bin/env python
# coding: utf-8

# In[1]:


empty, eos, maxlend, maxlenh, maxlen, seed = 0,1, 25, 10, 35, 42

activation_rnn_size = 40 if maxlend else 0
nb_unknown_words = 10

# function names
FN0 = 'vocabulary-embedding'  # filename of vocab embeddings
FN1 = 'train'  # filename of model weights

# training variables
seed = 42
optimizer = 'adam'
p_W, p_U, p_dense, p_emb, weight_decay = 0, 0, 0, 0, 0
regularizer = None


# In[2]:


from os import path
import _pickle as pickle
from collections import Counter
import numpy as np
import os
from sklearn.model_selection import train_test_split
import _pickle as pickle
from os import path
import random
import json
import pickle
import h5py
import numpy as np
import keras.backend as K
import argparse
import os
import time
import random
import argparse
import json

import numpy as np
from keras.callbacks import TensorBoard
import random

import numpy as np
from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.core import Lambda
import keras.backend as K
import numpy as np
from keras.models import load_model

import Levenshtein
import numpy as np
import random
from keras.preprocessing import sequence

desc_data_loc = "/home/daivatpc/Desktop/BTP 2 AS/NL NN/summary/sumdata/train/train.article.txt"
head_data_loc = "/home/daivatpc/Desktop/BTP 2 AS/NL NN/summary/sumdata/train/train.title.txt"


with open(desc_data_loc, 'r', encoding='utf-8') as f:
    desc_lines = f.read().split('\n')
with open(head_data_loc, 'r', encoding='utf-8') as f:
    head_lines = f.read().split('\n')
    
X_data, Y_data = [],[]
for i in range(len(desc_lines)):
    if( len(desc_lines[i].split()) <= maxlend and len(head_lines[i].split()) <= maxlenh-1 ):
        X_data.append(desc_lines[i].lower())
        Y_data.append(head_lines[i].lower())
print(len(X_data))


with open('tokens.pkl', 'wb') as fp:
            pickle.dump((Y_data[:500000],X_data[:500000]), fp, 2)


# In[3]:


"""Generate intial word embedding for headlines and description.

The embedding is limited to a fixed vocabulary size (`vocab_size`) but
a vocabulary of all the words that appeared in the data is built.
"""
import io

# static vars
FN = 'vocabulary-embedding'
seed = 42
vocab_size = 40000
embedding_dim = 100
lower = False

# index words
empty = 0  # RNN mask of no data
eos = 1  # end of sentence
start_idx = eos + 1  # first real word

# set random seed
np.random.seed(seed)


def build_vocab(lst):
    """Return vocabulary for iterable `lst`."""
    vocab_count = Counter(w for txt in lst for w in txt.split())
    vocab = list(map(lambda x: x[0], sorted(vocab_count.items(), key=lambda x: -x[1])))
    return vocab, vocab_count


def load_text():
    """Return vocabulary for pickled headlines and descriptions."""
    # read tokenized headlines and descriptions
    with open('tokens.pkl', 'rb') as fp:
        headlines, desc = pickle.load(fp)

    # map headlines and descriptions to lower case
    if lower:
        headlines = [h.lower() for h in headlines]
        desc = [h.lower() for h in desc]

    return headlines, desc


def print_most_popular_tokens(vocab):
    """Print th most popular tokens in vocabulary dictionary `vocab`."""
    print('Most popular tokens:')
    print(vocab[:50])
    print('Total vocab size: {:,}'.format(len(vocab)))

'''
def plot_word_distributions(vocab, vocab_count):
    """Plot word distribution in headlines and discription."""
    plt.plot([vocab_count[w] for w in vocab])
    plt.gca().set_xscale("log", nonposx='clip')
    plt.gca().set_yscale("log", nonposy='clip')
    title = 'word distribution in headlines and discription'
    plt.title(title)
    plt.xlabel('rank')
    plt.ylabel('total appearances')
    plt.savefig(path.join(config.path_outputs, '{}.png'.format(title)))
'''


def get_idx(vocab):
    """Add empty and end-of-sentence tokens to vocabulary and return tuple (vocabulary, reverse-vocabulary)."""
    word2idx = dict((word, idx + start_idx) for idx, word in enumerate(vocab))
    word2idx['<empty>'] = empty
    word2idx['<eos>'] = eos
    idx2word = dict((idx, word) for word, idx in word2idx.items())
    return word2idx, idx2word


def get_glove():
    """Load GloVe embedding weights and indices."""
    glove_name = 'glove.6B.{}d.txt'.format(embedding_dim)
    glove_n_symbols = 0
    with io.open(glove_name, encoding="utf-8") as glovedata :
        for line in glovedata:
            glove_n_symbols+=1
    #glove_n_symbols = sum(1 for line in open(glove_name))
    print('{:,} GloVe symbols'.format(glove_n_symbols))

    # load embedding weights and index dictionary
    glove_index_dict = {}
    glove_embedding_weights = np.empty((glove_n_symbols, embedding_dim))
    globale_scale = .1
    with io.open(glove_name, encoding="utf-8") as fp:
        i = 0
        for l in fp:
            l = l.strip().split()
            w = l[0]
            glove_index_dict[w] = i
            glove_embedding_weights[i, :] = list(map(float, l[1:]))
            i += 1
    glove_embedding_weights *= globale_scale
    print('GloVe std dev: {:.4f}'.format(glove_embedding_weights.std()))

    # add lower case version of the keys to the dict
    for w, i in glove_index_dict.items():
        w = w.lower()
        if w not in glove_index_dict:
            glove_index_dict[w] = i

    return glove_embedding_weights, glove_index_dict


def initialize_embedding(vocab_size, embedding_dim, glove_embedding_weights):
    """Use GloVe to initialize random embedding matrix with same scale as glove."""
    shape = (vocab_size, embedding_dim)
    scale = glove_embedding_weights.std() * np.sqrt(12) / 2  # uniform and not normal
    embedding = np.random.uniform(low=-scale, high=scale, size=shape)
    print('random-embedding/glove scale: {:.4f} std: {:.4f}'.format(scale, embedding.std()))
    return embedding


def copy_glove_weights(embedding, idx2word, glove_embedding_weights, glove_index_dict):
    """Copy from glove weights of words that appear in our short vocabulary (idx2word)."""
    c = 0
    for i in range(vocab_size):
        w = idx2word[i]
        g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        if g is None and w.startswith('#'):  # glove has no hastags (I think...)
            w = w[1:]
            g = glove_index_dict.get(w, glove_index_dict.get(w.lower()))
        if g is not None:
            embedding[i, :] = glove_embedding_weights[g, :]
            c += 1
    print('number of tokens, in small vocab: {:,} found in glove and copied to embedding: {:.4f}'.format(c, c / float(vocab_size)))
    return embedding


def build_word_to_glove(embedding, word2idx, idx2word, glove_index_dict, glove_embedding_weights):
    """Map full vocabulary to glove based on cosine distance."""
    glove_thr = 0.5
    word2glove = {}
    for w in word2idx:
        if w in glove_index_dict:
            g = w
        elif w.lower() in glove_index_dict:
            g = w.lower()
        elif w.startswith('#') and w[1:] in glove_index_dict:
            g = w[1:]
        elif w.startswith('#') and w[1:].lower() in glove_index_dict:
            g = w[1:].lower()
        else:
            continue
        word2glove[w] = g

    # for every word outside the embedding matrix find the closest word inside the emmbedding matrix.
    # Use cos distance of GloVe vectors.
    # Allow for the last `nb_unknown_words` words inside the embedding matrix to be considered to be outside.
    # Dont accept distances below `glove_thr`
    normed_embedding = embedding / np.array(
        [np.sqrt(np.dot(gweight, gweight)) for gweight in embedding])[:, None]

    nb_unknown_words = 100

    glove_match = []
    for w, idx in word2idx.items():
        if idx >= vocab_size - nb_unknown_words and w.isalpha() and w in word2glove:
            gidx = glove_index_dict[word2glove[w]]
            gweight = glove_embedding_weights[gidx, :].copy()

            # find row in embedding that has the highest cos score with gweight
            gweight /= np.sqrt(np.dot(gweight, gweight))
            score = np.dot(normed_embedding[:vocab_size - nb_unknown_words], gweight)
            while True:
                embedding_idx = score.argmax()
                s = score[embedding_idx]
                if s < glove_thr:
                    break
                if idx2word[embedding_idx] in word2glove:
                    glove_match.append((w, embedding_idx, s))
                    break
                score[embedding_idx] = -1

    glove_match.sort(key=lambda x: -x[2])
    print()
    print('# of GloVe substitutes found: {:,}'.format(len(glove_match)))

    # manually check that the worst substitutions we are going to do are good enough
    for orig, sub, score in glove_match[-10:]:
        print('{:.4f}'.format(score), orig, '=>', idx2word[sub])

    # return a lookup table of index of outside words to index of inside words
    return dict((word2idx[w], embedding_idx) for w, embedding_idx, _ in glove_match)


def to_dense_vector(word2idx, corpus, description, bins=50):
    """Create a dense vector representation of headlines."""
    data = [[word2idx[token] for token in txt.split()] for txt in corpus]
    #plt.hist(list(map(len, data)), bins=bins)
    #plt.savefig(path.join(config.path_outputs, '{}_distribution.png'.format(description)))
    return data


def summarize_vocab(vocab, vocab_count):
    """Print the most popular tokens and plot token distributions."""
    print_most_popular_tokens(vocab)
    #plot_word_distributions(vocab, vocab_count)


def main():
    """Generate intial word embedding for headlines and description."""
    headlines, desc = load_text()  # load headlines and descriptions
    vocab, vocab_count = build_vocab(headlines + desc)  # build vocabulary
    summarize_vocab(vocab, vocab_count)  # summarize vocabulary
    word2idx, idx2word = get_idx(vocab)  # add special tokens and get reverse vocab lookup
    glove_embedding_weights, glove_index_dict = get_glove()  # load GloVe data

    # initialize embedding
    embedding = initialize_embedding(vocab_size, embedding_dim, glove_embedding_weights)
    embedding = copy_glove_weights(embedding, idx2word, glove_embedding_weights, glove_index_dict)

    # map vocab to GloVe using cosine similarity
    glove_idx2idx = build_word_to_glove(embedding, word2idx, idx2word, glove_index_dict, glove_embedding_weights)

    # create a dense vector representation of headlines and descriptions
    description_vector = to_dense_vector(word2idx, desc, 'description')
    headline_vector = to_dense_vector(word2idx, headlines, 'headline')

    # write vocabulary to disk
    with open('{}.pkl'.format(FN), 'wb') as fp:
        pickle.dump((embedding, idx2word, word2idx, glove_idx2idx), fp, 2)

    # write data to disk
    with open('{}.data.pkl'.format(FN), 'wb') as fp:
        pickle.dump((description_vector, headline_vector), fp, 2)

if __name__ == '__main__':
    main()


# In[4]:


"""Utility methods."""

def join_ingredients(ingredients_listlist):
    """Join multiple lists of ingredients with ' , '."""
    return [' , '.join(i) for i in ingredients_listlist]


def get_flat_ingredients_list(ingredients_joined_train):
    """Flatten lists of ingredients encoded as a string into a single list."""
    return ' , '.join(ingredients_joined_train).split(' , ')


def section_print():
    """Memorized function keeping track of section number."""
    section_number = 0

    def inner(message):
        """Print section number."""
        global section_number
        section_number += 1
        print('Section {}: {}'.format(section_number, message))
    print('Section {}: initializing section function'.format(section_number))
    return inner


def is_filename_char(x):
    """Return True if x is an acceptable filename character."""
    if x.isalnum():
        return True
    if x in ['-', '_']:
        return True
    return False


def url_to_filename(filename):
    """Map a URL string to filename by removing unacceptable characters."""
    return "".join(x for x in filename if is_filename_char(x))


def prt(label, word_idx, idx2word):
    """Map `word_idx` list to words and print it with its associated `label`."""
    words = [idx2word[word] for word in word_idx]
    print('{}: {}\n'.format(label, ' '.join(words)))


def str_shape(x):
    """Format the dimension of numpy array `x` as a string."""
    return 'x'.join([str(element) for element in x.shape])


def load_embedding(nb_unknown_words):
    """Read word embeddings and vocabulary from disk."""
    with open('{}.pkl'.format(FN0), 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
    vocab_size, embedding_size = embedding.shape
    print('dimension of embedding space for words: {:,}'.format(embedding_size))
    print('vocabulary size: {:,} the last {:,} words can be used as place holders for unknown/oov words'.
          format(vocab_size, nb_unknown_words))
    print('total number of different words: {:,}'.format(len(idx2word)))
    print('number of words outside vocabulary which we can substitue using glove similarity: {:,}'.
          format(len(glove_idx2idx)))
    print('number of words that will be regarded as unknonw(unk)/out-of-vocabulary(oov): {:,}'.
          format(len(idx2word) - vocab_size - len(glove_idx2idx)))
    return embedding, idx2word, word2idx, glove_idx2idx


def load_data():
    """Read recipe data from disk."""
    with open('{}.data.pkl'.format(FN0), 'rb') as fp:
        X, Y = pickle.load(fp)
    print('number of examples', len(X), len(Y))
    return X, Y


def process_vocab(idx2word, vocab_size, oov0, nb_unknown_words):
    """Update vocabulary to account for unknown words."""
    # reserve vocabulary space for unkown words
    for i in range(nb_unknown_words):
        idx2word[vocab_size - 1 - i] = '<{}>'.format(i)

    # mark words outside vocabulary with ^ at their end
    for i in range(oov0, len(idx2word)):
        idx2word[i] = idx2word[i] + '^'

    # add empty word and end-of-sentence to vocab
    idx2word[empty] = '_'
    idx2word[eos] = '~'

    return idx2word


def load_split_data(nb_val_samples, seed):
    """Create train-test split."""
    # load data and create train test split
    X, Y = load_data()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=nb_val_samples, random_state=seed)
    del X, Y  # free up memory by removing X and Y
    return X_train, X_test, Y_train, Y_test


if __name__ == '__main__':
    print(url_to_filename('http://stackoverflow.com/questions/295135/turn-a-string-into-a-valid-filename'))


# In[5]:


"""Create an LSTM model for recipe summarization."""

def inspect_model(model):
    """Print the structure of Keras `model`."""
    for i, l in enumerate(model.layers):
        print(i, 'cls={} name={}'.format(type(l).__name__, l.name))
        weights = l.get_weights()
        print_str = ''
        for weight in weights:
            print_str += str_shape(weight) + ' '
        print(print_str)
        print()


class SimpleContext(Lambda):
    """Class to implement `simple_context` method as a Keras layer."""

    def __init__(self, fn, rnn_size, **kwargs):
        """Initialize SimpleContext."""
        self.rnn_size = rnn_size
        super(SimpleContext, self).__init__(fn, **kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        """Compute mask of maxlend."""
        return input_mask[:, maxlend:]

    def get_output_shape_for(self, input_shape):
        """Get output shape for a given `input_shape`."""
        nb_samples = input_shape[0]
        n = 2 * (self.rnn_size - activation_rnn_size)
        return (nb_samples, maxlenh, n)


def create_model(vocab_size, embedding_size, LR, rnn_layers, rnn_size, embedding=None):
    """Construct and compile LSTM model."""
    # create a standard stacked LSTM
    if embedding is not None:
        embedding = [embedding]
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_size,
                        input_length=maxlen,
                        W_regularizer=regularizer, dropout=p_emb, weights=embedding, mask_zero=True,
                        name='embedding_1'))
    for i in range(rnn_layers):
        lstm = LSTM(rnn_size, return_sequences=True,
                    W_regularizer=regularizer, U_regularizer=regularizer,
                    b_regularizer=regularizer, dropout_W=p_W, dropout_U=p_U,
                    name='lstm_{}'.format(i + 1))
        model.add(lstm)
        model.add(Dropout(p_dense, name='dropout_{}'.format(i + 1)))

    def simple_context(X, mask, n=activation_rnn_size):
        """Reduce the input just to its headline part (second half).

        For each word in this part it concatenate the output of the previous layer (RNN)
        with a weighted average of the outputs of the description part.
        In this only the last `rnn_size - activation_rnn_size` are used from each output.
        The first `activation_rnn_size` output is used to compute the weights for the averaging.
        """
        desc, head = X[:, :maxlend, :], X[:, maxlend:, :]
        head_activations, head_words = head[:, :, :n], head[:, :, n:]
        desc_activations, desc_words = desc[:, :, :n], desc[:, :, n:]

        # RTFM http://deeplearning.net/software/theano/library/tensor/basic.html#theano.tensor.batched_tensordot
        # activation for every head word and every desc word
        activation_energies = K.batch_dot(head_activations, desc_activations, axes=(2, 2))
        # make sure we dont use description words that are masked out
        if mask is not None:
            param = 1. - K.cast(mask[:, :maxlend], 'float32')
            activation_energies = activation_energies + -1e20 * K.expand_dims(param, 1)

        # for every head word compute weights for every desc word
        activation_energies = K.reshape(activation_energies, (-1, maxlend))
        activation_weights = K.softmax(activation_energies)
        activation_weights = K.reshape(activation_weights, (-1, maxlenh, maxlend))

        # for every head word compute weighted average of desc words
        desc_avg_word = K.batch_dot(activation_weights, desc_words, axes=(2, 1))
        return K.concatenate((desc_avg_word, head_words))

    if activation_rnn_size:
        model.add(SimpleContext(simple_context, rnn_size, name='simplecontext_1'))

    model.add(TimeDistributed(Dense(
        vocab_size,
        W_regularizer=regularizer,
        b_regularizer=regularizer,
        name='time_distributed_2')))
    model.add(Activation('softmax', name='activation_1'))

    # opt = Adam(lr=LR)  # keep calm and reduce learning rate
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    K.set_value(model.optimizer.lr, np.float32(LR))
    return model


# In[6]:


"""Generate samples.

Variation on https://github.com/ryankiros/skip-thoughts/blob/master/decoding/search.py
"""

def lpadd(x):
    """Left (pre) pad a description to maxlend and then add eos.

    The eos is the input to predicting the first word in the headline
    """
    assert maxlend >= 0
    if maxlend == 0:
        return [eos]
    n = len(x)
    if n > maxlend:
        x = x[-maxlend:]
        n = maxlend
    return [empty] * (maxlend - n) + x + [eos]


def beamsearch(
        predict, start, k, maxsample, use_unk, empty, temperature, nb_unknown_words,
        vocab_size, model, batch_size, avoid=None, avoid_score=1):
    """Return k samples (beams) and their NLL scores, each sample is a sequence of labels.

    All samples starts with an `empty` label and end with `eos` or truncated to length of `maxsample`.
    You need to supply `predict` which returns the label probability of each sample.
    `use_unk` allow usage of `oov` (out-of-vocabulary) label in samples.
    """
    def sample(energy, n, temperature=temperature):
        """Sample at most n elements according to their energy."""
        n = min(n, len(energy))
        prb = np.exp(-np.array(energy) / temperature)
        res = []
        for i in range(n):
            z = np.sum(prb)
            r = np.argmax(np.random.multinomial(1, prb / z, 1))
            res.append(r)
            prb[r] = 0.  # make sure we select each element only once
        return res

    dead_samples = []
    dead_scores = []
    live_k = 1  # samples that did not yet reached eos
    live_samples = [list(start)]
    live_scores = [0]

    while live_k:
        # for every possible live sample calc prob for every possible label
        probs = predict(live_samples, empty=empty, model=model, batch_size=batch_size)

        # total score for every sample is sum of -log of word prb
        cand_scores = np.array(live_scores)[:, None] - np.log(probs)
        cand_scores[:, empty] = 1e20
        if not use_unk:
            for i in range(nb_unknown_words):
                cand_scores[:, vocab_size - 1 - i] = 1e20

        if avoid:
            for a in avoid:
                for i, s in enumerate(live_samples):
                    n = len(s) - len(start)
                    if n < len(a):
                        # at this point live_sample is before the new word,
                        # which should be avoided, is added
                        cand_scores[i, a[n]] += avoid_score

        live_scores = list(cand_scores.flatten())

        # find the best (lowest) scores we have from all possible dead samples and
        # all live samples and all possible new words added
        scores = dead_scores + live_scores
        ranks = sample(scores, k)
        n = len(dead_scores)
        ranks_dead = [r for r in ranks if r < n]
        ranks_live = [r - n for r in ranks if r >= n]

        dead_scores = [dead_scores[r] for r in ranks_dead]
        dead_samples = [dead_samples[r] for r in ranks_dead]

        live_scores = [live_scores[r] for r in ranks_live]

        # append the new words to their appropriate live sample
        voc_size = probs.shape[1]
        live_samples = [live_samples[r // voc_size] + [r % voc_size] for r in ranks_live]

        # live samples that should be dead are...
        # even if len(live_samples) == maxsample we dont want it dead because we want one
        # last prediction out of it to reach a headline of maxlenh
        zombie = [s[-1] == eos or len(s) > maxsample for s in live_samples]

        # add zombies to the dead
        dead_samples += [s for s, z in zip(live_samples, zombie) if z]
        dead_scores += [s for s, z in zip(live_scores, zombie) if z]
        # remove zombies from the living
        live_samples = [s for s, z in zip(live_samples, zombie) if not z]
        live_scores = [s for s, z in zip(live_scores, zombie) if not z]
        live_k = len(live_samples)

    return dead_samples + live_samples, dead_scores + live_scores


def keras_rnn_predict(samples, empty, model, batch_size):
    """For every sample, calculate probability for every possible label.

    You need to supply your RNN model and maxlen - the length of sequences it can handle
    """
    sample_lengths = list(map(len, samples))
    assert all(l > maxlend for l in sample_lengths)
    assert all(l[maxlend] == eos for l in samples)
    # pad from right (post) so the first maxlend will be description followed by headline
    data = sequence.pad_sequences(samples, maxlen=maxlen, value=empty, padding='post', truncating='post')
    probs = model.predict(data, verbose=0, batch_size=batch_size)
    return np.array([prob[sample_length - maxlend - 1]
                     for prob, sample_length in zip(probs, sample_lengths)])


def vocab_fold(xs, oov0, glove_idx2idx, vocab_size, nb_unknown_words):
    """Convert list of word indices that may contain words outside vocab_size to words inside.

    If a word is outside, try first to use glove_idx2idx to find a similar word inside.
    If none exist then replace all accurancies of the same unknown word with <0>, <1>, ...
    """
    xs = [x if x < oov0 else glove_idx2idx.get(x, x) for x in xs]
    # the more popular word is <0> and so on
    outside = sorted([x for x in xs if x >= oov0])
    # if there are more than nb_unknown_words oov words then put them all in nb_unknown_words-1
    outside = dict((x, vocab_size - 1 - min(i, nb_unknown_words - 1)) for i, x in enumerate(outside))
    xs = [outside.get(x, x) for x in xs]
    return xs


def vocab_unfold(desc, xs, oov0):
    """Covert a description to a list of word indices."""
    # assume desc is the unfolded version of the start of xs
    unfold = {}
    for i, unfold_idx in enumerate(desc):
        fold_idx = xs[i]
        if fold_idx >= oov0:
            unfold[fold_idx] = unfold_idx
    return [unfold.get(x, x) for x in xs]


def gensamples(
        skips, short, data, idx2word, oov0, glove_idx2idx, vocab_size,
        nb_unknown_words, avoid=None, avoid_score=1, **kwargs):
    """Generate text samples."""
    # unpack data
    X, Y = data

    # if data is full dataset pick a random header and description
    if not isinstance(X[0], int):
        i = random.randint(0, len(X) - 1)
        x = X[i]
        y = Y[i]
    else:
        x = X
        y = Y

    # print header and description
    print('HEAD:', ' '.join(idx2word[w] for w in y[:maxlenh]))
    print('DESC:', ' '.join(idx2word[w] for w in x[:maxlend]))

    if avoid:
        # avoid is a list of avoids. Each avoid is a string or list of word indeicies
        if isinstance(avoid, str) or isinstance(avoid[0], int):
            avoid[avoid]
        avoid = [a.split() if isinstance(a, str) else a for a in avoid]
        avoid = [[a] for a in avoid]

    print('HEADS:')
    samples = []
    if maxlend == 0:
        skips = [0]
    else:
        skips = range(min(maxlend, len(x)), max(maxlend, len(x)), abs(maxlend - len(x)) // skips + 1)
    for s in skips:
        start = lpadd(x[:s])
        fold_start = vocab_fold(start, oov0, glove_idx2idx, vocab_size, nb_unknown_words)
        sample, score = beamsearch(
            predict=keras_rnn_predict,
            start=fold_start,
            maxsample=maxlen,
            empty=empty,
            nb_unknown_words=nb_unknown_words,
            vocab_size=vocab_size,
            avoid=avoid,
            **kwargs
        )
        assert all(s[maxlend] == eos for s in sample)
        samples += [(s, start, scr) for s, scr in zip(sample, score)]

    samples.sort(key=lambda x: x[-1])
    codes = []
    for sample, start, score in samples:
        code = ''
        words = []
        sample = vocab_unfold(start, sample, oov0)[len(start):]
        for w in sample:
            if w == eos:
                break
            words.append(idx2word[w])
            code += chr(w // (256 * 256)) + chr((w // 256) % 256) + chr(w % 256)

        if short:
            distance = min([100] + [-Levenshtein.jaro(code, c) for c in codes])
            if distance > -0.6:
                print(score, ' '.join(words))
        else:
                print(score, ' '.join(words))
        codes.append(code)
    return samples


# In[7]:


"""Data generator generates batches of inputs and outputs/labels for training.

The inputs are each made from two parts. The first maxlend words are the original description, followed by `eos` followed by the headline which we want to predict, except for the last word in the headline which is always `eos` and then `empty` padding until `maxlen` words.

For each, input, the output is the headline words (without the start `eos` but with the ending `eos`) padded with `empty` words up to `maxlenh` words. The output is also expanded to be y-hot encoding of each word.

To be more realistic, the second part of the input should be the result of generation and not the original headline.
Instead we will flip just `nflips` words to be from the generator, but even this is too hard and instead
implement flipping in a naive way (which consumes less time.) Using the full input (description + eos + headline) generate predictions for outputs. Faor nflips random words from the output, replace the original word with the word with highest probability from the prediction.
"""

def flip_headline(x, nflips, model, debug, oov0, idx2word):
    """Flip some of the words in the second half (headline) with words predicted by the model."""
    if nflips is None or model is None or nflips <= 0:
        return x

    batch_size = len(x)
    assert np.all(x[:, maxlend] == eos)
    probs = model.predict(x, verbose=0, batch_size=batch_size)
    x_out = x.copy()
    for b in range(batch_size):
        # pick locations we want to flip
        # 0...maxlend-1 are descriptions and should be fixed
        # maxlend is eos and should be fixed
        flips = sorted(random.sample(range(maxlend + 1, maxlen), nflips))
        if debug and b < debug:
            print(b)
        for input_idx in flips:
            if x[b, input_idx] == empty or x[b, input_idx] == eos:
                continue
            # convert from input location to label location
            # the output at maxlend (when input is eos) is feed as input at maxlend+1
            label_idx = input_idx - (maxlend + 1)
            prob = probs[b, label_idx]
            w = prob.argmax()
            if w == empty:  # replace accidental empty with oov
                w = oov0
            if debug and b < debug:
                print('{} => {}'.format(idx2word[x_out[b, input_idx]], idx2word[w]),)
            x_out[b, input_idx] = w
        if debug and b < debug:
            print()
    return x_out


def conv_seq_labels(xds, xhs, nflips, model, debug, oov0, glove_idx2idx, vocab_size, nb_unknown_words, idx2word):
    """Convert description and hedlines to padded input vectors; headlines are one-hot to label."""
    batch_size = len(xhs)
    assert len(xds) == batch_size
    x = [
        vocab_fold(lpadd(xd) + xh, oov0, glove_idx2idx, vocab_size, nb_unknown_words)
        for xd, xh in zip(xds, xhs)]  # the input does not have 2nd eos
    x = sequence.pad_sequences(x, maxlen=maxlen, value=empty, padding='post', truncating='post')
    x = flip_headline(x, nflips=nflips, model=model, debug=debug, oov0=oov0, idx2word=idx2word)

    y = np.zeros((batch_size, maxlenh, vocab_size))
    for i, xh in enumerate(xhs):
        xh = vocab_fold(xh, oov0, glove_idx2idx, vocab_size, nb_unknown_words) + [eos] + [empty] * maxlenh  # output does have a eos at end
        xh = xh[:maxlenh]
        y[i, :, :] = np_utils.to_categorical(xh, vocab_size)

    return x, y


def gen(Xd, Xh, batch_size, nb_batches, nflips, model, debug, oov0, glove_idx2idx, vocab_size, nb_unknown_words, idx2word):
    """Yield batches.

    for training use nb_batches=None
    for validation generate deterministic results repeating every nb_batches
    """
    # while training it is good idea to flip once in a while the values of the headlines from the
    # value taken from Xh to value generated by the model.
    c = nb_batches if nb_batches else 0
    while True:
        xds = []
        xhs = []
        if nb_batches and c >= nb_batches:
            c = 0
        new_seed = random.randint(0, 2e10)
        random.seed(c + 123456789 + seed)
        for b in range(batch_size):
            t = random.randint(0, len(Xd) - 1)

            xd = Xd[t]
            s = random.randint(min(maxlend, len(xd)), max(maxlend, len(xd)))
            xds.append(xd[:s])

            xh = Xh[t]
            s = random.randint(min(maxlenh, len(xh)), max(maxlenh, len(xh)))
            xhs.append(xh[:s])

        # undo the seeding before we yield inorder not to affect the caller
        c += 1
        random.seed(new_seed)

        yield conv_seq_labels(
            xds,
            xhs,
            nflips=nflips,
            model=model,
            debug=debug,
            oov0=oov0,
            glove_idx2idx=glove_idx2idx,
            vocab_size=vocab_size,
            nb_unknown_words=nb_unknown_words,
            idx2word=idx2word,
        )


# In[8]:


"""Train a sequence to sequence model.


"""

epochs=15
rnn_size=512
rnn_layers=3
nsamples=640
nflips=0
temperature=.8
lr=0.001
warm_start='store_true'
batch_size=32

# set sample sizes
nb_train_samples = np.int(np.floor(nsamples / batch_size)) * batch_size  # num training samples
nb_val_samples = nb_train_samples  # num validation samples

# seed weight initialization
random.seed(seed)
np.random.seed(seed)

embedding, idx2word, word2idx, glove_idx2idx = load_embedding(nb_unknown_words)
vocab_size, embedding_size = embedding.shape
oov0 = vocab_size - nb_unknown_words
idx2word = process_vocab(idx2word, vocab_size, oov0, nb_unknown_words)
X_train, X_test, Y_train, Y_test = load_split_data(nb_val_samples, seed)

# print a sample recipe to make sure everything looks right
print('Random head, description:')
i = 811
prt('H', Y_train[i], idx2word)
prt('D', X_train[i], idx2word)

# save model initialization parameters
model_params = (dict(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    LR=lr,
    rnn_layers=rnn_layers,
    rnn_size=rnn_size,
))
with open('model_params.json', 'w') as f:
    json.dump(model_params, f)


model = create_model(
    vocab_size=vocab_size,
    embedding_size=embedding_size,
    LR=lr,
    embedding=embedding,
    rnn_layers=rnn_layers,
    rnn_size=rnn_size,
)
inspect_model(model)

# load pre-trained model weights
FN1_filename = '{}.hdf5'.format(FN1)
if warm_start and FN1 and os.path.exists(FN1_filename):
    model.load_weights(FN1_filename)
    print('Model weights loaded from {}'.format(FN1_filename))

# print samples before training
gensamples(
    skips=2,
    k=10,
    batch_size=batch_size,
    short=False,
    temperature=temperature,
    use_unk=True,
    model=model,
    data=(X_test, Y_test),
    idx2word=idx2word,
    oov0=oov0,
    glove_idx2idx=glove_idx2idx,
    vocab_size=vocab_size,
    nb_unknown_words=nb_unknown_words,
)

# get train and validation generators
r = next(gen(X_train, Y_train, batch_size=batch_size, nb_batches=None, nflips=None, model=None, debug=False, oov0=oov0, glove_idx2idx=glove_idx2idx, vocab_size=vocab_size, nb_unknown_words=nb_unknown_words, idx2word=idx2word))
traingen = gen(X_train, Y_train, batch_size=batch_size, nb_batches=None, nflips=nflips, model=model, debug=False, oov0=oov0, glove_idx2idx=glove_idx2idx, vocab_size=vocab_size, nb_unknown_words=nb_unknown_words, idx2word=idx2word)
valgen = gen(X_test, Y_test, batch_size=batch_size, nb_batches=nb_val_samples // batch_size, nflips=None, model=None, debug=False, oov0=oov0, glove_idx2idx=glove_idx2idx, vocab_size=vocab_size, nb_unknown_words=nb_unknown_words, idx2word=idx2word)

# define callbacks for training
callbacks = [TensorBoard(
    log_dir=str(time.time()),
    histogram_freq=0, write_graph=False, write_images=False)]

# train model and save weights
h = model.fit_generator(
    traingen, samples_per_epoch=nb_train_samples,
    #traingen, samples_per_epoch=3,
    nb_epoch=epochs, validation_data=valgen, nb_val_samples=nb_val_samples,
    #nb_epoch=1, validation_data=valgen, nb_val_samples=1,
    callbacks=callbacks,
)

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

model.save_weights(FN1_filename, overwrite=True)

# print samples after training
samples = gensamples(
    skips=2,
    k=10,
    batch_size=batch_size,
    short=False,
    temperature=temperature,
    use_unk=True,
    model=model,
    data=(X_test, Y_test),
    idx2word=idx2word,
    oov0=oov0,
    glove_idx2idx=glove_idx2idx,
    vocab_size=vocab_size,
    nb_unknown_words=nb_unknown_words,
)


# In[9]:


"""Predict a title for a recipe."""

# set seeds in random libraries
seed = 42
random.seed(seed)
np.random.seed(seed)


def load_weights(model, filepath):
    """Load all weights possible into model from filepath.

    This is a modified version of keras load_weights that loads as much as it can
    if there is a mismatch between file and model. It returns the weights
    of the first layer in which the mismatch has happened
    """
    print('Loading', filepath, 'to', model.name)
    with h5py.File(filepath, mode='r') as f:
        # new file format
        layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]

        # we batch weight value assignments in a single backend call
        # which provides a speedup in TensorFlow.
        weight_value_tuples = []
        for name in layer_names:
            print(name)
            g = f[name]
            weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
            if len(weight_names):
                weight_values = [g[weight_name] for weight_name in weight_names]
                try:
                    layer = model.get_layer(name=name)
                except:
                    layer = None
                if not layer:
                    print('failed to find layer', name, 'in model')
                    print('weights', ' '.join(str_shape(w) for w in weight_values))
                    print('stopping to load all other layers')
                    weight_values = [np.array(w) for w in weight_values]
                    break
                symbolic_weights = layer.trainable_weights + layer.non_trainable_weights
                weight_value_tuples += zip(symbolic_weights, weight_values)
                weight_values = None
        K.batch_set_value(weight_value_tuples)
    return weight_values


def main(sample_str=None):
    '''
    """Predict a title for a recipe."""
    # load model parameters used for training
    with open('model_params.json', 'r') as f:
        model_params = json.load(f)

    # create placeholder model
    model = create_model(**model_params)

    # load weights from training run
    load_weights(model, '{}.hdf5'.format(FN1))
    '''
    # load recipe titles and descriptions
    with open('vocabulary-embedding.data.pkl', 'rb') as fp:
        X_data, Y_data = pickle.load(fp)

    # load vocabulary
    with open('{}.pkl'.format(FN0), 'rb') as fp:
        embedding, idx2word, word2idx, glove_idx2idx = pickle.load(fp)
    vocab_size, embedding_size = embedding.shape
    oov0 = vocab_size - nb_unknown_words

    if sample_str is None:
        # load random recipe description if none provided
        i = np.random.randint(len(X_data))
        sample_str = ''
        sample_title = ''
        for w in X_data[i]:
            sample_str += idx2word[w] + ' '
        for w in Y_data[i]:
            sample_title += idx2word[w] + ' '
        y = Y_data[i]
        print('Randomly sampled recipe:')
        print(sample_title)
        print(sample_str)
    else:
        sample_title = ''
        y = [eos]

    x = [word2idx[w.rstrip('^')] for w in sample_str.split()]

    samples = gensamples(
        skips=2,
        k=1,
        batch_size=2,
        short=False,
        temperature=1.,
        use_unk=True,
        model=model,
        data=(x, y),
        idx2word=idx2word,
        oov0=oov0,
        glove_idx2idx=glove_idx2idx,
        vocab_size=vocab_size,
        nb_unknown_words=nb_unknown_words,
    )

    headline = samples[0][0][len(samples[0][1]):]
    ' '.join(idx2word[w] for w in headline)

if __name__ == '__main__':
    main(sample_str=X_data[5002])


# In[10]:


X_data[5001]


# In[11]:


main(X_data[1])


# In[12]:


main(None)


# In[13]:


# train model and save weights
#h = model.fit_generator(
#    traingen, samples_per_epoch=nb_train_samples,
    #traingen, samples_per_epoch=3,
#    nb_epoch=epochs, validation_data=valgen, nb_val_samples=nb_val_samples,
    #nb_epoch=1, validation_data=valgen, nb_val_samples=1,
#    callbacks=callbacks,
#)

# model.save('my_model.h5')  # creates a HDF5 file 'my_model.h5'

#model.save_weights(FN1_filename, overwrite=True)


# In[14]:


main(X_data[5002])


# In[ ]:


main(None)


# In[ ]:


main(None)


# In[15]:


model.summary()


# In[16]:


test_list = [100,200,300,400,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100,2200,2300,2400,2500,2600,2700,2800,2900,3000]
for i in map(lambda x: x+1, test_list):
    try:
        main(X_data[i])
    except IndexError:
        i+=1


# In[23]:


main(X_data[40000])


# In[ ]:




