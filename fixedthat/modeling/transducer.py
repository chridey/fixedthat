import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

from fixedthat.modeling.customDecoderRNN import CustomDecoderRNN
from fixedthat.modeling.state import HiddenState
    
class Transducer(CustomDecoderRNN):
    #1) need to keep track of the index into the encoder outputs
    #2) need to update this index when we output a state change
    
    def __init__(self, source_vocab, target_vocab, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False,
            max_count=50, count_hidden_size=15):
        super(Transducer, self).__init__(source_vocab, target_vocab, max_len, hidden_size,
                                         sos_id, eos_id,
                                         n_layers, rnn_cell, bidirectional,
                                         input_dropout_p, dropout_p, use_attention,
                                         max_count, count_hidden_size)
        
        self.transducer_rnn = self.rnn_cell(2*hidden_size+count_hidden_size,
                                            hidden_size,
                                            n_layers, batch_first=True, dropout=dropout_p)        
        self.rnn = self.rnn_cell(2*hidden_size,
                                            hidden_size,
                                            n_layers, batch_first=True, dropout=dropout_p)        

        self.eob_id = self.target_vocab.stoi['<e>']
        self.specials |= {self.source_vocab.stoi['<e>']}
        
    def _validate_args(self, encoder_inputs, decoder_inputs, encoder_hidden,
                       encoder_outputs, function, teacher_forcing_ratio,
                       counts=None):        
        
        decoder_inputs, batch_size, max_length, extras = super(Transducer, self)._validate_args(encoder_inputs,
                                                                                                decoder_inputs,
                                                                             encoder_hidden, encoder_outputs,
                                                                             function, teacher_forcing_ratio,
                                                                             counts)

        #enhance the input to include the encoder index at each time step
        encoder_index = decoder_inputs[:, :, 0].data.eq(self.eob_id).long()
        x = [torch.zeros(decoder_inputs.size(0), 1).long()]
        if torch.cuda.is_available():
            x[0] = x[0].cuda()
        #print(x[0], encoder_index[:,0])
        
        for i in range(decoder_inputs.size(1)-1):
            x.append(x[-1] + encoder_index[:,i].contiguous().view(-1,1))
        x = Variable(torch.cat(x, dim=1))
        decoder_inputs = torch.cat([decoder_inputs, x.unsqueeze(2)], dim=2)

        if torch.cuda.is_available():
            decoder_inputs = decoder_inputs.cuda()
        '''
        print(x[0])
        print(decoder_inputs[0, :, 0])
        print([self.target_vocab.itos[i] for i in decoder_inputs[0, :, 0].data.numpy()])
        print(self.eob_id)
        '''

        lengths, counters, encoder_input_lookup, encoder_input_data = extras
        counters = torch.cat([counters.unsqueeze(2), x[:,0].contiguous().view(-1,1,1)], dim=2)
        extras = [lengths, counters, encoder_input_lookup, encoder_input_data]
                    
        return decoder_inputs, batch_size, max_length, extras        

    def _init_state(self, encoder_hidden):
        #return initial states for transducer and decoder and context
        hidden = super(Transducer, self)._init_state(encoder_hidden)
        return HiddenState(hidden, None, None)
    
    def forward_step(self, input_var, hidden, encoder_outputs, function):
        #input_var also includes the current encoder index        

        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        input_size = encoder_outputs.size(1)
        
        embedded = self.embedding(input_var[:, :, 0])
        embedded = self.input_dropout(embedded)

        count_embedded = self.count_embedding(input_var[:, :, 1])
        count_embedded = self.input_dropout(count_embedded)

        embedded = torch.cat([embedded, count_embedded], dim=2)

        decoder_hidden, transducer_hidden, previous_context = hidden.fields

        #get the current encoder index
        encoder_index = input_var[:, :, 2].contiguous().view(-1)
        offset = Variable((torch.LongTensor(range(batch_size))*input_size).view(-1,1).expand(batch_size, output_size).contiguous().view(batch_size*output_size), requires_grad=False)
        if torch.cuda.is_available():
            offset = offset.cuda()

        '''
        print(input_var.shape)
        print(encoder_index.shape)
        print(encoder_outputs.shape)
        print(offset.shape)
        print(encoder_index.max())
        print(offset.max())            
        '''
            
        new_context = torch.index_select(encoder_outputs.contiguous().view(batch_size*input_size, -1),
                                         0, offset + torch.clamp(encoder_index,
                                                               max=input_size-1)).view(batch_size, output_size,
                                                                         encoder_outputs.size(2))
        #print(new_context.shape)
        if previous_context is None:
            zeros = Variable(torch.zeros(batch_size, 1, encoder_outputs.size(2)))
            if torch.cuda.is_available():
                zeros = zeros.cuda()
            context = torch.cat([zeros, new_context], dim=1)
        else:
            context = torch.cat([previous_context.transpose(0,1), new_context], dim=1)

        #print(context.shape, embedded.shape)
        transducer_input = torch.cat([context[:, :-1, :], embedded], dim=2)
        transducer_output, transducer_hidden = self.transducer_rnn(transducer_input, transducer_hidden)

        #for now, don't use attention b/c we have a small window size
        #attn = None
        #if self.use_attention:
        #    output, attn = self.attention(output, encoder_outputs)

        #now concatenate all the contexts to the transducer output
        decoder_input = torch.cat([transducer_output, context[:, 1:, ]], dim=2)
        output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
        
        predicted_softmax = function(self.out(output.contiguous().view(-1,
                                                                       self.hidden_size))).view(batch_size,
                                                                                                       output_size,
                                                                                                       -1)

        #if div by count
        #predicted_softmax = predicted_softmax / (10-input_var[:, :, 1].unsqueeze(2).float())
                
        return predicted_softmax, HiddenState(decoder_hidden, transducer_hidden,
                                              context[:, 1:, :].transpose(0,1)), None #context needs same shape as hidden states, which are seq length first


    def decode(self, step, decoder_outputs, sequence_symbols, lengths=None,
               counters=None, encoder_input_lookup=None, encoder_input_data=None, 
               use_teacher_forcing=True, k=1, filter_illegal=True, verbose=False):

        if counters is not None:
            encoder_index = counters[:, :, 1]
            counters = counters[:, :, 0]
        extras = [lengths, counters, encoder_input_lookup, encoder_input_data]
        sequence_symbols, input_vars, extras = super(Transducer, self).decode(step, decoder_outputs,
                                                                              sequence_symbols, *extras,
                                                                              use_teacher_forcing=use_teacher_forcing,
                                                                              k=k, filter_illegal=filter_illegal,
                                                                              verbose=verbose)

        #for all outputs where the sequence symbols are EOB, update the index in input_vars
        if input_vars is not None:
            encoder_index = encoder_index + sequence_symbols[-1].eq(self.eob_id).long()
            input_vars = torch.cat([input_vars, encoder_index.unsqueeze(2)], dim=2)
            #TODO: the input_vars should be less than the length of the encoder input

            #print(encoder_index.data.cpu().numpy().tolist())
            lengths, counters = extras[:2]
            #print(counters, encoder_index)        
            counters = torch.cat([counters.unsqueeze(2), encoder_index.unsqueeze(2)], dim=2)
            extras = [lengths, counters] + extras[2:]
        
        return sequence_symbols, input_vars, extras
