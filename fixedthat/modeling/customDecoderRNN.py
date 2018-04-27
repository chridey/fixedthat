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

from fixedthat.preprocessing import ftfy_utils as ut
    
#from seq2seq.models.attention import Attention    
from seq2seq.models import DecoderRNN

from fixedthat.modeling.customAttention import Attention

class CustomDecoderRNN(DecoderRNN):
    r"""
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        hidden_size (int): the number of features in the hidden state `h`
        sos_id (int): index of the start of sentence symbol
        eos_id (int): index of the end of sentence symbol
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        dropout_p (float, optional): dropout probability for the output sequence (default: 0)
        use_attention(bool, optional): flag indication whether to use attention mechanism or not (default: false)
    Attributes:
        KEY_ATTN_SCORE (str): key used to indicate attention weights in `ret_dict`
        KEY_LENGTH (str): key used to indicate a list representing lengths of output sequences in `ret_dict`
        KEY_SEQUENCE (str): key used to indicate a list of sequences in `ret_dict`
    Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **inputs** (batch, seq_len, input_size): list of sequences, whose length is the batch size and within which
          each sequence is a list of token IDs.  It is used for teacher forcing when provided. (default `None`)
        - **encoder_hidden** (num_layers * num_directions, batch_size, hidden_size): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, hidden_size): tensor with containing the outputs of the encoder.
          Used for attention mechanism (default is `None`).
        - **function** (torch.nn.Module): A function used to generate symbols from RNN hidden state
          (default is `torch.nn.functional.log_softmax`).
        - **teacher_forcing_ratio** (float): The probability that teacher forcing will be used. A random number is
          drawn uniformly from 0-1 for every decoding token, and if the sample is smaller than the given value,
          teacher forcing would be used (default is 0).
    Outputs: decoder_outputs, decoder_hidden, ret_dict
        - **decoder_outputs** (seq_len, batch, vocab_size): list of tensors with size (batch_size, vocab_size) containing
          the outputs of the decoding function.
        - **decoder_hidden** (num_layers * num_directions, batch, hidden_size): tensor containing the last hidden
          state of the decoder.
        - **ret_dict**: dictionary containing additional information as follows {*KEY_LENGTH* : list of integers
          representing lengths of output sequences, *KEY_SEQUENCE* : list of sequences, where each sequence is a list of
          predicted token IDs }.
    """

    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'

    def __init__(self, source_vocab, target_vocab, max_len, hidden_size,
            sos_id, eos_id,
            n_layers=1, rnn_cell='gru', bidirectional=False,
            input_dropout_p=0, dropout_p=0, use_attention=False,
            max_count=50, count_hidden_size=15, copy_predictor=False):
        super(CustomDecoderRNN, self).__init__(len(target_vocab), max_len, hidden_size,
                input_dropout_p, dropout_p,
                n_layers, rnn_cell)

        if copy_predictor is not None and not use_attention :
            print('WARNING: copy_predictor was provided but attention was set to false so copy_predictor is not used')
        self.source_vocab = source_vocab
        self.target_vocab = target_vocab
        
        self.bidirectional_encoder = bidirectional
        self.rnn = self.rnn_cell(hidden_size+count_hidden_size,
                                 hidden_size,
                                 n_layers, batch_first=True, dropout=dropout_p)

        self.output_size = len(target_vocab)
        self.max_length = max_len

        self.max_count = max_count
        self.count_hidden_size = count_hidden_size
        
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)

        if self.count_hidden_size:
            self.count_embedding = nn.Embedding(self.max_count, self.count_hidden_size)

        self.copy_predictor = copy_predictor
            
        if use_attention:
            self.attention = Attention(self.hidden_size, copy_predictor)
                
            #self.out = nn.Linear(2*self.hidden_size, self.output_size)
        #else:
        self.out = nn.Linear(self.hidden_size, self.output_size)

        self.copy = False
        if self.copy:
            self.gen = nn.Linear(3*self.hidden_size, 1)
        
        #remove some characters like . that are "free" ie if not in input we can generate them
        self.specials = {self.source_vocab.stoi[i] for i in (ut.stopwords | set('~!@#$%^&*()_+`-={}|[]\\:";\'<>?,./') | set(j for j in self.target_vocab.stoi if all(map(lambda x: not x.isalnum(), j))))} | {0}
        self.target_specials = {self.target_vocab.stoi[self.source_vocab.itos[i]] for i in self.specials} - {0}
        
    def forward_step(self, input_var, hidden, encoder_outputs, function, mask=None, copy_mask=None):
        #print(input_var)
        
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        embedded = self.embedding(input_var[:, :, 0])
        embedded = self.input_dropout(embedded)

        if self.count_hidden_size:
            count_embedded = self.count_embedding(input_var[:, :, 1])
            count_embedded = self.input_dropout(count_embedded)

            embedded = torch.cat([embedded, count_embedded], dim=2)
        
        output, hidden = self.rnn(embedded, hidden)

        attn = None
        if self.use_attention:                                             
            output, attn = self.attention(output, encoder_outputs, mask, copy_mask)
        #print(encoder_outputs, attn)
            
        predicted_softmax = function(self.out(output.contiguous().view(-1,
                                                                       self.hidden_size))).view(batch_size,
                                                                                                       output_size,
                                                                                                       -1)
        if False: #self.copy:
            output = torch.cat([output, embedded], dim=2)
            p_gen = F.sigmoid(self.gen(output.view(-1, 3*self.hidden_size))).view(batch_size, output_size)
        
        return predicted_softmax, hidden, attn

    def decode(self, step, decoder_outputs, sequence_symbols=None, lengths=None,
               counters=None, encoder_input_lookup=None, encoder_input_data=None, 
               use_teacher_forcing=True, k=1, filter_illegal=True, verbose=False):

        if not use_teacher_forcing:
            symbols = []
            batch_size = decoder_outputs.size(0) // k
            vocab_size = decoder_outputs.size(1)
            
            decoder_outputs = decoder_outputs.view(batch_size, -1)
            top_scores, top_symbols = decoder_outputs.topk(2*k, dim=1)
            #scores = scores.data.cpu().numpy()
            top_symbols = top_symbols.data.cpu().numpy()
            current_counters = counters.data.cpu().contiguous().view(-1, k).numpy()

            if verbose:
                print(batch_size)
                print(vocab_size)
                print(k)
                print([q for q in top_scores[0].data.cpu().numpy()])
                print([q % vocab_size for q in top_symbols[0]])
                print('tokens', [self.target_vocab.itos[q % vocab_size] for q in top_symbols[0]])
                print(current_counters[0])

            try:
                target_specials = self.target_specials
            except AttributeError:    
                target_specials = {self.target_vocab.stoi[self.source_vocab.itos[i]] for i in self.specials} - {0}
            #top_symbols = decoder_outputs[-1].topk(2)[1].data.cpu().view(-1,2).numpy()
            #current_counters = counters.data.cpu().view(-1).numpy()
            
            '''
            print('sym shape', top_symbols.shape)
            print('ctr shape', current_counters.shape)

            idx = 0
            print('ctrs', current_counters[idx])                    
            print('symbols', top_symbols[idx])
            print('tokens', [self.target_vocab.itos[q] for q in top_symbols[idx]])
            legal = [self.eos_id] + [self.target_vocab.stoi[self.source_vocab.itos[i]] for i in encoder_inputs_data[idx]]
            print('legal', legal)
            legal = Variable(torch.LongTensor(legal)).cuda()
            print('legal scores', decoder_outputs[-1][idx][legal])
            print('top legal', decoder_outputs[-1][idx][legal].topk(1))
            symbol = int(legal[decoder_outputs[-1][idx][legal].topk(1)[1]].data.cpu()[0])
            print('symbol', symbol)
            '''

            ref = []
            new_scores = []
            for idx, candidates in enumerate(top_symbols):
                #TODO: modify vocabulary distribution to zero out illegal choices?
                #if the counter is > 0, legal symbols are vocab - {EOS}, need to ignore EOS so take top 2
                good = []
                good_scores = []

                legal = {self.eos_id} | {self.target_vocab.stoi[self.source_vocab.itos[i]] for i in encoder_input_data[idx]} | target_specials
                scores = top_scores[idx]
                multiplier = 2
                while True:
                    for sidx, candidate in enumerate(candidates):
                        symbol = candidate % vocab_size
                        predecessor = candidate / vocab_size
                        count = current_counters[idx][predecessor]
                        #print(sidx, candidate, symbol, predecessor, count, scores[sidx])
                        if count > 0:
                            if filter_illegal and symbol == self.eos_id:
                                continue                            
                        
                            source_symbol = self.source_vocab.stoi[self.target_vocab.itos[symbol]]
                            if source_symbol not in encoder_input_lookup[idx] and source_symbol not in self.specials:
                                count -= 1
                        #if the counter is at 0, legal symbols are the input and EOS and the specials (find highest scoring)
                            
                        #allow UNK?
                        elif filter_illegal and symbol not in legal:
                            continue
                        good.append([symbol, predecessor, count])
                        good_scores.append(scores[sidx].data[0])
                    
                    if len(good) >= k:
                        good = good[:k]
                        good_scores = good_scores[:k]
                        break
                    #otherwise we have to go to the GPU and get the next top whatever, which could get expensive
                    #TODO - make this faster
                    multiplier *= 2
                    scores, candidates = decoder_outputs[idx].topk(multiplier*k)
                    scores = scores[(multiplier/2)*k:]
                    candidates = candidates[(multiplier/2)*k:].data.cpu().numpy()

                    if verbose:
                        print(scores)
                        print(candidates)
                        print('extra tokens', multiplier, [self.target_vocab.itos[q % vocab_size] for q in candidates])
                                        
                ref.append(good)
                new_scores.extend(good_scores)

            if verbose:
                print(ref[0])
            #print(good_scores)
            ref = Variable(torch.LongTensor(ref))
            if torch.cuda.is_available():
                ref = ref.cuda()

            #print(ref.shape)
            symbols = ref[:,:,0].contiguous().view(-1, 1)
            predecessors = ref[:,:,1].contiguous().view(-1,1)
            current_counters = ref[:,:,2].contiguous().view(-1,1)
            
            scores = Variable(torch.FloatTensor(new_scores)).view(-1,1)
            if torch.cuda.is_available():
                scores = scores.cuda()

            decoder_input = torch.cat([symbols.unsqueeze(2), current_counters.unsqueeze(2)], dim=2)
            #NOTE: if we have counters instead of current_counters, counters is never incremented and this is why we had that bug earlier
        else:
            symbols = decoder_outputs.topk(1)[1]

        if sequence_symbols is not None:
            sequence_symbols.append(symbols)

        if lengths is not None:
            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)

        if not use_teacher_forcing:
            extras = [lengths, current_counters, encoder_input_lookup, encoder_input_data]

            if k > 1:
                extras += [scores, predecessors]

            return sequence_symbols, decoder_input, extras

        return sequence_symbols, None, [lengths, None,None,None]
    
    def forward(self, encoder_inputs, decoder_inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0, counts=None, mask=None, copy_mask=None,
                    filter_illegal=True, use_prefix=False):
        #print(encoder_inputs)
        
        ret_dict = dict()
        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        decoder_inputs, batch_size, max_length, extras = self._validate_args(encoder_inputs, decoder_inputs,
                                                                             encoder_hidden, encoder_outputs,
                                                                             function, teacher_forcing_ratio,
                                                                             counts, use_prefix=use_prefix)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
        #print(use_teacher_forcing)
        #print(decoder_inputs.shape, encoder_inputs.shape)
                
        decoder_outputs = []
        sequence_symbols = []

        # Manual unrolling is used to support random teacher forcing.
        # If teacher_forcing_ratio is True or False instead of a probability, the unrolling can be done in graph
        #print(inputs.shape)
        if use_teacher_forcing:
            decoder_input = decoder_inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs,
                                                                     function=function,
                                                                     mask=mask, copy_mask=copy_mask)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                    
                decoder_outputs.append(step_output)
                if self.use_attention:
                    ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
                    
                sequence_symbols, _, extras = self.decode(di, decoder_outputs[-1], sequence_symbols, *extras,
                                                           use_teacher_forcing=use_teacher_forcing,
                                                           filter_illegal=filter_illegal)
        else:
            decoder_input = decoder_inputs[:, 0].unsqueeze(1)

            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden,
                                                                              encoder_outputs,
                                                                              function=function,
                                                                              mask=mask, copy_mask=copy_mask)
                step_output = decoder_output.squeeze(1)
                
                decoder_outputs.append(step_output)
                if self.use_attention:
                    ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)

                sequence_symbols, decoder_input, extras =self.decode(di, decoder_outputs[-1], sequence_symbols,
                                                                  *extras, use_teacher_forcing=use_teacher_forcing,
                                                                  filter_illegal=filter_illegal)
                if di == 0 and use_prefix:
                    decoder_input = decoder_inputs[:, 1].unsqueeze(1)
                                                                  
                #print(counters.max(), counters.min())

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = extras[0].tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, hidden_size) -> (#layers, #batch, #directions * hidden_size)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, encoder_inputs, decoder_inputs, encoder_hidden,
                       encoder_outputs, function, teacher_forcing_ratio,
                       counts=None, use_prefix=False):

        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")
            
        # inference batch size
        if decoder_inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if decoder_inputs is not None:
                batch_size = decoder_inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        encoder_input_data = encoder_inputs.data.cpu().numpy()
        encoder_input_lookup = [set(x) for x in encoder_input_data]

        extras = [encoder_input_data, encoder_input_lookup]
                                    
        # set default input and max decoding length
        if decoder_inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            if counts is None:
                raise ValueError("counts must be provided when inputs is None")

            inp = [self.sos_id]
            if use_prefix:
                assert(type(use_prefix) in (str, unicode))
                inp.append(self.target_vocab.stoi[use_prefix])

            decoder_inputs = Variable(torch.LongTensor(inp * batch_size),
                                    volatile=True).view(batch_size, 1+(use_prefix!=False))
            if torch.cuda.is_available():
                decoder_inputs = decoder_inputs.cuda()
                
            decoder_inputs = torch.cat([decoder_inputs.unsqueeze(2),
                                        counts.view(batch_size, 1).expand_as(decoder_inputs).unsqueeze(2)], dim=2)
            max_length = self.max_length
        else:
            #generate counts from decoder/encoder input pairs
                
            max_length = decoder_inputs.size(1) - 1 # minus the start of sequence symbol

            decoder_input_data = decoder_inputs.data.cpu().numpy()
            
            input_counts = []
            input_attention_map = []
            max_len_map = 0            
            for idx,x in enumerate(decoder_input_data):
                '''
                print([self.source_vocab.itos[symbol] for symbol in tmp[idx]])
                print(tmp[idx])
                print([self.target_vocab.itos[symbol] for symbol in x])
                print([self.source_vocab.stoi[self.target_vocab.itos[symbol]] for symbol in x])
                print([self.target_vocab.stoi[self.target_vocab.itos[symbol]] for symbol in x])
                print(list(x))
                '''
                
                #for ptr-gen, also need to map encoder input positions to vocabulary items
                #map index in target vocab with index into attention distribution over input along with count
                encoder_input_lookup = {}
                oov_idx = len(self.target_vocab.stoi)
                for position,symbol in enumerate(encoder_input_data[idx]):
                    if symbol not in encoder_input_lookup:
                        target_symbol = self.target_vocab.stoi[self.source_vocab.itos[symbol]]
                        if target_symbol == 0:
                            target_symbol = oov_idx
                            oov_idx += 1
                        encoder_input_lookup[symbol] = [target_symbol, position, 1]
                    else:
                        encoder_input_lookup[symbol][-1] += 1
                #find cases where symbol appears in both source and target (or just source) but is not in target vocab
                if len(encoder_input_lookup) > max_len_map:
                    max_len_map = len(encoder_input_lookup.values())
                input_attention_map.append(encoder_input_lookup.values())
                    
                count = 0
                source_symbols = []
                for symbol in x:
                    source_symbol = self.source_vocab.stoi[self.target_vocab.itos[symbol]]
                    source_symbols.append(source_symbol)
                    if source_symbol not in encoder_input_lookup and source_symbol not in self.specials:
                        count += 1
                        
                counts = []
                #print(count)
                for source_symbol in source_symbols:
                    if source_symbol not in encoder_input_lookup and source_symbol not in self.specials:
                        count -= 1
                    counts.append(count)
                #print(counts)
                
                input_counts.append(counts)
            input_counts = Variable(torch.LongTensor(input_counts))
            if torch.cuda.is_available():
                input_counts = input_counts.cuda()
                    
            decoder_inputs = torch.cat([decoder_inputs.unsqueeze(2),
                                        input_counts.unsqueeze(2)], dim=2)

            '''
            for i in range(len(input_attention_map)):
                pad = [max_oov+len(self.target_vocab.stoi), 0, 0]
                input_attention_map[i].extend(pad * (len(input_attention_map[i]))
            '''
        #need to determine if a symbol we generate is in the input
        extras = [np.array([max_length] * batch_size), #lengths
                  decoder_inputs[:,0,1].contiguous().view(-1, 1)] + extras

        return decoder_inputs, batch_size, max_length, extras
                
