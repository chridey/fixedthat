import unicodedata
import collections

import numpy as np
import torch
from torch.autograd import Variable
use_cuda = torch.cuda.is_available()

SOS_token = 0
EOS_token = 1
UNK_token = 2
EMPTY_token = 3

class Vocab:
    def __init__(self, min_count=5):
        self.min_count = min_count
        self._word2index = {}
        self.word2count = collections.Counter()
        self.index2word = {0: "SOS", 1: "EOS", 2: "UNK", 3: "EMPTY"}

    @property
    def size(self):
        return len(self.index2word)
    
    def word2index(self, word):
        if word in self._word2index:
            return self._word2index[word]
        return UNK_token
    
    def add_sentence(self, sentence):
        for word in sentence:
            self.add_word(word)
            
    def add_word(self, word):
        if self.word2count[word] >= self.min_count and word not in self._word2index:
            self._word2index[word] = len(self.index2word)
            self.index2word[len(self.index2word)] = word
        else:
            self.word2count[word] += 1

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicode2ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode2ascii(unicode(s, encoding='utf-8').strip()) #.lower().strip())
    #s = re.sub(r"([.!?])", r" \1", s)
    #s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.replace('COPY-REDUCE', 'COPY REDUCE')

def read_data(parent, ftfy, count=False):

    parents = []
    ftfys = []
    vocab = Vocab()

    print("Reading lines...")    
    with open(parent) as f:
        for line in f:
            line = normalize_string(line.strip()).split(' ')
            vocab.add_sentence(line)
            parents.append(line)

    print('finished parent')
    print('vocab size is %d' % vocab.size)
    print('Reading lines...')
    counts = collections.Counter()
    with open(ftfy) as f:
        for line in f:
            line = normalize_string(line.strip()).split(' ')
            if count:
                num_subs = 0
                for word in line:
                    if word not in {'SHIFT', 'REDUCE', 'COPY', 'COPY-REDUCE'}:
                        num_subs += 1
                ftfys.append(num_subs)
            else:
                vocab.add_sentence(line)
                ftfys.append(line)
            for word in line:
                counts[word] += 1

    weights = np.zeros(vocab.size) + 1
    for word in counts:
        weights[vocab.word2index(word)] += 1.
    weights = Variable(torch.FloatTensor(weights.min() / weights))
    if use_cuda:
        weights = weights.cuda()
                            
    print('finished ftfy')
    print('vocab size is %d' % vocab.size)
    
    return parents, ftfys, vocab, weights 


def indexesFromSentence(vocab, sentence):
    return [vocab.word2index(word) for word in sentence]

def variableFromSentence(vocab, sentence):
    indexes = indexesFromSentence(vocab, sentence)
    indexes.append(EOS_token)
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    
    return result

def variablesFromPair(vocab, pair):
    input_variable = variableFromSentence(vocab, pair[0])
    target_variable = variableFromSentence(vocab, pair[1])
    return (input_variable, target_variable)

def variablesFromCount(vocab, pair):
    input_variable = variableFromSentence(vocab, pair[0])
    target_variable = Variable(torch.FloatTensor([pair[1]]))
    if use_cuda:
        target_variable = target_variable.cuda()
    return (input_variable, target_variable)
