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
from utils import *
from encoderRNN import EncoderRNN
from stackBasedDecoder import StackBasedDecoder

from ftfy.preprocessing.transition_utils import transitions2ftfy

teacher_forcing_ratio = 0.5
MAX_LENGTH = 30

def get_valid_prediction(decoder_pred, stack, input_buffer, vocab):
    topv, topi = decoder_pred.data.topk(5)
    nis = [int(v) for v in topi[0]]

    valid = {vocab.word2index('SHIFT'): len(input_buffer) > 0,
             vocab.word2index('REDUCE'): len(stack) > 0,
             vocab.word2index('COPY'): len(stack) > 0,
             vocab.word2index('COPY-REDUCE'): len(stack) > 0,
             EOS_token: len(input_buffer)==0}

    for ni in nis:
        if ni not in valid or valid[ni]:
            return ni

def update_state(ni, vocab, stack, input_buffer, decoder_output, decoder_hidden, encoder2, input_variable):
    if vocab.index2word[ni] == 'SHIFT':
        stack.append(input_buffer.pop(0))
    elif vocab.index2word[ni] == 'REDUCE':
        stack = []
    elif vocab.index2word[ni] == 'COPY-REDUCE':
        for idx,item in stack:
            decoder_output, decoder_hidden = encoder2(input_variable[idx],
                                                     decoder_hidden)
            stack = []
    elif vocab.index2word[ni] == 'COPY':
        idx, item = stack[-1]
        decoder_output, decoder_hidden = encoder2(input_variable[idx],
                                                  decoder_hidden)
    else:
        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_output, decoder_hidden = encoder2(decoder_input,
                                                 decoder_hidden)
        
    return stack, input_buffer, decoder_output, decoder_hidden

def update_counts(target, decoder_pred, expected_counts, total_counts, vocab):
    if expected_counts is None:
        #expected_counts = Variable(torch.zeros(3))
        #if use_cuda:
        #    expected_counts = expected_counts.cuda()
        expected_counts = torch.exp(decoder_pred[(0,0,0), (vocab.word2index('SHIFT'),
                                                vocab.word2index('REDUCE'),
                                                vocab.word2index('COPY'))])
    else:
        expected_counts = expected_counts + torch.exp(decoder_pred[(0,0,0), (vocab.word2index('SHIFT'),
                                                vocab.word2index('REDUCE'),
                                                vocab.word2index('COPY'))])
        
    ni = int(target.data[0])
    if total_counts is None:
        total_counts = Variable(torch.zeros(3))
        if use_cuda:
            total_counts = total_counts.cuda()    

    if vocab.index2word[ni] == 'SHIFT':
        increment = [1,0,0]
    elif vocab.index2word[ni] == 'REDUCE':
        increment = [0,1,0]
    elif vocab.index2word[ni] == 'COPY':
        increment = [0,0,1]
    else:
        increment = [0,0,0]
        
    increment = Variable(torch.FloatTensor(increment))
    if use_cuda:
        increment = increment.cuda()
        
    return expected_counts, total_counts + increment

def train(input_variable, target_variable, encoder, decoder,
          encoder_optimizer, decoder_optimizer, criterion, vocab, max_length=MAX_LENGTH, encoder2=None):

    if encoder2 is None:
        encoder2 = encoder
    
    encoder_hidden = encoder.initHidden()
        
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    for ei in range(min(input_length, max_length)):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    if use_cuda:
        decoder_input = decoder_input.cuda()
    decoder_hidden = encoder_hidden #encoder.initHidden() #
        
    use_teacher_forcing = True if np.random.random() < teacher_forcing_ratio else False

    stack = []
    input_buffer = list(enumerate(encoder_outputs[i] for i in range(min(input_length, max_length))))
    decoder_output, decoder_hidden = encoder2(decoder_input,
                                              decoder_hidden)        

    expected_counts = None
    total_counts = None
        
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_pred = decoder(stack,
                                   input_buffer,
                                   decoder_output)

            expected_counts, total_counts = update_counts(target_variable[di],
                                                          decoder_pred, expected_counts,
                                                          total_counts, vocab)
            #print(expected_counts, total_counts)
            loss += criterion(decoder_pred, target_variable[di]) + nn.MSELoss(size_average=False)(expected_counts,
                                                                                                  total_counts)
            decoder_input = target_variable[di]  # Teacher forcing
            
            ni = int(decoder_input.data[0])
            if vocab.index2word[ni] == 'SHIFT':
                if len(input_buffer):
                    stack.append(input_buffer.pop(0))
                else:
                    break
            elif vocab.index2word[ni] == 'REDUCE':
                stack = []
            elif vocab.index2word[ni] == 'COPY-REDUCE':
                for idx,item in stack:
                    decoder_output, decoder_hidden = encoder2(input_variable[idx],
                                                             decoder_hidden)
                stack = []
            elif vocab.index2word[ni] == 'COPY':
                idx, item = stack[-1]
                decoder_output, decoder_hidden = encoder2(input_variable[idx],
                                                        decoder_hidden)
            else:
                decoder_output, decoder_hidden = encoder2(decoder_input,
                                                         decoder_hidden)
    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_pred = decoder(stack,
                                    input_buffer,
                                    decoder_output)

            ni = get_valid_prediction(decoder_pred, stack, input_buffer, vocab)
            
            #loss += criterion(decoder_pred, target_variable[di])
            expected_counts, total_counts = update_counts(target_variable[di],
                                                          decoder_pred, expected_counts, total_counts, vocab)
            loss += criterion(decoder_pred, target_variable[di]) + nn.MSELoss()(expected_counts, total_counts)

            if ni == EOS_token:
                break

            stack, input_buffer, decoder_output, decoder_hidden = update_state(ni,
                                                                               vocab,
                                                                               stack,
                                                                               input_buffer,
                                                                               decoder_output,
                                                                               decoder_hidden,
                                                                               encoder2,
                                                                               input_variable)
                
    loss.backward()
    
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def trainIters(pairs, vocab, encoder, decoder, print_every=1000, plot_every=100, learning_rate=0.01, weight=None,
               encoder2=None):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    training_pairs = [variablesFromPair(vocab, pairs[i]) for i in np.random.choice(len(pairs), len(pairs), False)]
        
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(weight=weight)
    n_iters = len(pairs)+1
    for iter in range(1, n_iters):
        training_pair = training_pairs[iter - 1]
        input_variable = training_pair[0]
        target_variable = training_pair[1]
    
        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion, vocab, encoder2=encoder2)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, 1. * iter / n_iters),
                                         iter, 1. * iter / n_iters * 100, print_loss_avg))

            evaluateRandomly(pairs, vocab, encoder, decoder, n=5, encoder2=encoder2)
                
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    #showPlot(plot_losses)                                                                                          

def evaluate(vocab, encoder, decoder, sentence, max_length=MAX_LENGTH, encoder2=None):
    if encoder2 is None:
        encoder2 = encoder
        
    input_variable = variableFromSentence(vocab, sentence)
    input_length = input_variable.size()[0]
    
    encoder_hidden = encoder.initHidden()
    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(min(input_length, max_length)):
        encoder_output, encoder_hidden = encoder(input_variable[ei],
                                                 encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
    if use_cuda:
        decoder_input = decoder_input.cuda()

    decoder_hidden = encoder_hidden #encoder.initHidden() #

    transitions = []
    stack = []
    input_buffer = list(enumerate(encoder_outputs[i] for i in range(min(input_length, max_length))))
    
    decoder_output, decoder_hidden = encoder2(decoder_input,
                                             decoder_hidden)
    
    while len(transitions) < 4*max_length:
        decoder_pred = decoder(stack,
                               input_buffer,
                               decoder_output)
        
        ni = get_valid_prediction(decoder_pred, stack, input_buffer, vocab)
        if ni == EOS_token:
            transitions.append('<EOS>')
            break
        
        stack, input_buffer, decoder_output, decoder_hidden = update_state(ni,
                                                                           vocab,
                                                                           stack,
                                                                           input_buffer,
                                                                           decoder_output,
                                                                           decoder_hidden,
                                                                           encoder2,
                                                                           input_variable)
                
        transitions.append(vocab.index2word[ni])
        
    return transitions

def evaluateRandomly(pairs, vocab, encoder, decoder, n=10, encoder2=None):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        print('==', transitions2ftfy(pair[1], pair[0], ignore_invalid=True))
        output_transitions = evaluate(vocab, encoder, decoder, pair[0], encoder2=encoder2)
        print('<', ' '.join(output_transitions))
        print('<<', ' '.join(transitions2ftfy(output_transitions, pair[0], ignore_invalid=True)))
        print('')

def main(train_prefix, test_prefix):

    train_parent, train_ftfy, vocab, weight = read_data(train_prefix + '.parent', train_prefix + '.ftfy')
    test_parent, test_ftfy, _, _ = read_data(test_prefix + '.parent', test_prefix + '.ftfy')    
        
    hidden_size = 256
    encoder1 = EncoderRNN(vocab.size, hidden_size)
    encoder2 = EncoderRNN(vocab.size, hidden_size)
    attn_decoder1 = StackBasedDecoder(hidden_size, hidden_size, vocab.size)

    if use_cuda:
        encoder1 = encoder1.cuda()
        encoder2 = encoder2.cuda()
        attn_decoder1 = attn_decoder1.cuda()

    weight = None
    for i in range(5):
        trainIters(zip(train_parent, train_ftfy), vocab, encoder1, attn_decoder1, print_every=10000, weight=weight, encoder2=encoder2)

if __name__ == '__main__':
    train_prefix = sys.argv[1]
    test_prefix = sys.argv[2]

    main(train_prefix, test_prefix)
