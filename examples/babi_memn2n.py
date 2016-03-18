from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Lambda, Dense
from keras.layers.recurrent import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from functools import reduce
import tarfile
import numpy as np
import re

from seya.layers.memnn2 import MemN2N


def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true, only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories, and then convert the sentences into a single story.
    If max_length is supplied, any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    return data


def vectorize_stories(data, word_idx, memory_length, input_length):
    X = []
    Xq = []
    Y = []
    for stories, query, answer in data:
        x = [[word_idx[w] for w in story] for story in stories]
        xq = [word_idx[w] for w in query]
        y = np.zeros(len(word_idx) + 1)  # let's not forget that index 0 is reserved
        y[word_idx[answer]] = 1
        X.append(x)
        Xq.append(xq)
        Y.append(y)
    padded_X = [pad_sequences(xx, maxlen=input_length) for xx in X]
    out = []
    # pad stories
    for story in padded_X:
        Z = np.zeros((memory_length, input_length))
        Z[-len(story):] = story
        out.append(Z)
    return (np.asarray(out),
            pad_sequences(Xq, maxlen=input_length), np.array(Y))

path = get_file('babi-tasks-v1-2.tar.gz',
                origin='http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2.tar.gz')
tar = tarfile.open(path)

challenges = {
    # QA1 with 10,000 samples
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt',
    # QA2 with 10,000 samples
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt',
}

# challenge_type = 'two_supporting_facts_10k'
challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
train_stories = get_stories(tar.extractfile(challenge.format('train')))
test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = sorted(
    reduce(
        lambda x, y: x | y, (
            set(
                reduce(lambda x, y: x + y, stories) + q + [answer])
            for stories, q, answer in train_stories + test_stories)))

vocab_size = len(vocab) + 1
memory_length = max(map(len, (x for x, _, _ in train_stories + test_stories)))
# query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))
input_length = max(map(len, (x[0] for x, _, _ in train_stories + test_stories)))
word_idx = dict((c, i+1) for i, c in enumerate(vocab))
output_dim = 64

inputs_train, queries_train, answers_train = vectorize_stories(
    train_stories, word_idx, memory_length, input_length)
inputs_test, queries_test, answers_test = vectorize_stories(
    test_stories, word_idx, memory_length, input_length)

queries_train = queries_train[:, None, :]
queries_test = queries_test[:, None, :]

# Model definition
facts = Sequential()
facts.add(Lambda(lambda x: x, input_shape=(memory_length, vocab_size),
                 output_shape=(memory_length, vocab_size)))
question = Sequential()
question.add(Lambda(lambda x: x, input_shape=(1, vocab_size),
                    output_shape=(1, vocab_size)))

memnn = MemN2N([facts, question], output_dim=output_dim, input_dim=vocab_size,
               input_length=input_length,
               memory_length=memory_length, hops=1, output_shape=(vocab_size,))
memnn.build()

model = Sequential()
model.add(memnn)
model.add(LSTM(32))
model.add(Dense(vocab_size, activation="softmax"))

W = model.trainable_weights[0].get_value()
model.compile("rmsprop", "categorical_crossentropy")
model.fit([inputs_train, queries_train], answers_train,
          batch_size=32,
          nb_epoch=100,
          show_accuracy=True,
          validation_data=([inputs_test, queries_test], answers_test))

# print(W - model.trainable_weights[0].get_value())
