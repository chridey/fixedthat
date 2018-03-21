import collections

import sys
import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

from data import *
from encoderRNN import EncoderRNN
from feedforward import Feedforward

teacher_forcing_ratio = 0.5
MAX_LENGTH = 30

def get_encoder_outputs(encoder, input_variable, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()
        
    input_length = input_variable.size()[0]
    
    encoder_outputs = Variable(torch.zeros(min(input_length, max_length), encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(min(input_length, max_length)):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    return encoder_outputs

def train(input_variable, target_variable, encoder, classifier,
          optimizer, criterion, max_length=MAX_LENGTH):
    
    optimizer.zero_grad()
    encoder_outputs = get_encoder_outputs(encoder, input_variable)

    prediction = classifier(torch.mean(encoder_outputs, dim=0))
    loss = criterion(prediction, target_variable)                            

    loss.backward()
    
    optimizer.step()

    return loss.data[0]

import time
import math


def asMinutes(s):
    m = math.floor(1. * s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def trainIters(train_pairs, test_pairs,
               vocab, encoder, classifier, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    training_pairs = [variablesFromCount(vocab, train_pairs[i]) for i in np.random.choice(len(train_pairs),
                                                                                   len(train_pairs), False)]
        
    optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)

    criterion = nn.PoissonNLLLoss() #nn.MSELoss() #
    n_iters = len(train_pairs)+1
    for iter in range(1, n_iters):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
        
        loss = train(input_variable, target_variable, encoder, classifier, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, 1. * iter / n_iters),
                                         iter, 1. * iter / n_iters * 100, print_loss_avg))

            evaluateRandomly(test_pairs, vocab, encoder, classifier, n=5)
                
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)                                                                                          

def evaluate(vocab, encoder, classifier, sentence, max_length=MAX_LENGTH):

    input_variable = variableFromSentence(vocab, sentence)
    encoder_outputs = get_encoder_outputs(encoder, input_variable)
    prediction = classifier(torch.mean(encoder_outputs, dim=0))

    return prediction
    
def evaluateRandomly(pairs, vocab, encoder, classifier, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        prediction = evaluate(vocab, encoder, classifier, pair[0])
        print('<', torch.exp(prediction)) #prediction) #
        print('')

def main(train_prefix, test_prefix):

    train_parent, train_ftfy, vocab, _ = read_data(train_prefix + '.parent', train_prefix + '.ftfy', True)
    test_parent, test_ftfy, _, _ = read_data(test_prefix + '.parent', test_prefix + '.ftfy', True)    

    print(collections.Counter(train_ftfy))
    print(collections.Counter(test_ftfy))
    
    hidden_size = 256
    encoder = EncoderRNN(vocab.size, hidden_size, False)
    classifier = Feedforward(hidden_size, hidden_size, 1, link=None)

    if use_cuda:
        encoder = encoder.cuda()
        classifier = classifier.cuda()

    for i in range(5):
        trainIters(zip(train_parent, train_ftfy), zip(test_parent, test_ftfy),
                   vocab, encoder, classifier, print_every=10000)

if __name__ == '__main__':
    train_prefix = sys.argv[1]
    test_prefix = sys.argv[2]

    main(train_prefix, test_prefix)
