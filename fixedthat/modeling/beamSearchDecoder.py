'''
beam search adapted from https://github.com/IBM/pytorch-seq2seq
'''

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from seq2seq.models.TopKDecoder import TopKDecoder, _inflate
from fixedthat.modeling.state import HiddenState

class BeamSearchDecoder(TopKDecoder):
    '''
    beam search with additional functionality for handling the extra features (count, topic, constrained decoding described in Sections 4.2 and 4.4 of "Fixed That for You: Generating Contrastive Claims with Semantic Edits")
    '''
    
    def forward(self, encoder_inputs, decoder_inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, retain_output_probs=True, counts=None,
                mask=None, copy_mask=None, filter_illegal=True, use_prefix=False, features=None, cce_keywords=None):
        """
        Forward rnn for MAX_LENGTH steps.  Look at :func:`seq2seq.models.DecoderRNN.DecoderRNN.forward_rnn` for details.
        """

        decoder_inputs, batch_size, max_length, extras = self.rnn._validate_args(encoder_inputs, decoder_inputs,
                                                                                 encoder_hidden, encoder_outputs,
                                                                                 function, teacher_forcing_ratio,
                                                                                 counts, features=features, cce_keywords=cce_keywords)

        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)
        self.pos_index = self.pos_index.expand(batch_size, self.k).contiguous().view(batch_size*self.k,1)
        if torch.cuda.is_available():
            self.pos_index = self.pos_index.cuda()
            
        # Inflate the initial hidden states to be of size: b*k x h
        encoder_hidden = self.rnn._init_state(encoder_hidden)
        if encoder_hidden is None:
            hidden = None
        elif isinstance(encoder_hidden, HiddenState):
            hidden = []
            for field in encoder_hidden.fields:
                if field is None:
                    hidden.append(None)
                elif isinstance(field, tuple):
                    hidden.append(tuple([_inflate(h.unsqueeze(2), self.k, 2).view(1,batch_size*self.k,-1) for h in field]))
                else:
                    hidden.append(_inflate(field.unsqueeze(2), self.k, 2).view(1,batch_size*self.k,-1))
            hidden = HiddenState(*hidden)
        else:
            if isinstance(encoder_hidden, tuple):
                hidden = tuple([_inflate(h.unsqueeze(2), self.k, 2).view(1,batch_size*self.k,-1) for h in encoder_hidden])
            else:
                hidden = _inflate(encoder_hidden.unsqueeze(2), self.k, 2).view(1,batch_size*self.k,-1)

        # ... same idea for encoder_outputs and decoder_outputs
        if self.rnn.use_attention:
            inflated_encoder_outputs = _inflate(encoder_outputs.unsqueeze(1),self.k,1).view(batch_size*self.k,
                                                                                            encoder_outputs.size(1),
                                                                                            -1)
        else:
            inflated_encoder_outputs = None

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)
        if torch.cuda.is_available():
            sequence_scores = sequence_scores.cuda()
            
        # Initialize the input vector
        #input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))
        #inputs is B x 1 x C (1 or 2) -> B x K x C -> B * K x C
        #decoder_inputs = decoder_inputs[:, 0].unsqueeze(1).expand(batch_size, self.k, -1).view(batch_size*self.k,-1)

        lengths, counters = extras[:2]
        input_var = _inflate(decoder_inputs[:, 0].unsqueeze(1), self.k, 1).view(batch_size*self.k,1,-1)
        counters = _inflate(counters[:, 0].unsqueeze(1), self.k, 1).view(batch_size*self.k,1,-1)
        extras = [None, counters] + extras[2:]

        if features is not None:
            for idx in range(len(features)):
                features[idx] = _inflate(features[idx][:, 0].unsqueeze(1), self.k, 1).view(batch_size*self.k,1,-1)

        if mask is not None:
            mask = _inflate(mask.view(batch_size, 1, -1), self.k, 1).view(batch_size*self.k, -1)
        if copy_mask is not None:
            copy_mask = _inflate(copy_mask.view(batch_size, 1, -1), self.k, 1).view(batch_size*self.k, -1)
                
        if cce_keywords is not None:
            cce_keywords = _inflate(cce_keywords.view(batch_size, 1, -1), self.k, 1).view(batch_size*self.k, -1)

        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for step in range(0, max_length):
            # Run the RNN one step forward
            log_softmax_output, hidden, _ = self.rnn.forward_step(input_var, hidden,
                                                                  inflated_encoder_outputs, function=function,
                                                                  copy_mask=copy_mask, mask=mask, cce_keywords=cce_keywords)

            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output)

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = _inflate(sequence_scores, self.V, 1)
            sequence_scores += log_softmax_output.squeeze(1)

            #need to change the decode function to handle legal inputs instead of the code below
            sequence_symbols, input_var, extras = self.rnn.decode(step, sequence_scores, [],
                                                         *extras, use_teacher_forcing=False, k=self.k,
                                                                  filter_illegal=filter_illegal, verbose=False,
                                                                  features=features, cce_keywords=cce_keywords)
            scores, predecessors = extras[-2:]
            extras = extras[:-2]
            sequence_scores = scores.view(batch_size * self.k, 1)
            predecessors = (predecessors + self.pos_index)

            # Update fields for next timestep
            if isinstance(hidden, HiddenState):
                new_hidden = []
                for field in hidden.fields:
                    #print(field.shape)
                    if isinstance(field, tuple):
                        new_hidden.append(tuple([h.index_select(1, predecessors.squeeze()) for h in field]))
                    else:
                        new_hidden.append(field.index_select(1, predecessors.squeeze()))
                    
                hidden = HiddenState(*new_hidden)

                decoder_hidden = hidden.decoder_hidden
            else:
                if isinstance(hidden, tuple):
                    hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
                else:
                    hidden = hidden.index_select(1, predecessors.squeeze())
                decoder_hidden = hidden
                
            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            #eos_indices = input_var.data.eq(self.EOS)
            eos_indices = sequence_symbols[-1].data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(sequence_symbols[-1])
            stored_hidden.append(decoder_hidden)

        # Do backtracking to return the optimal values
        output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
                                                    stored_predecessors, stored_emitted_symbols,
                                                    stored_scores, batch_size, self.hidden_size)

        # Build return objects
        decoder_outputs = [step[:, 0, :] for step in output]
        if isinstance(h_n, tuple):
            decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
        else:
            decoder_hidden = h_n[:, :, 0, :]
        metadata = {}
        metadata['inputs'] = decoder_inputs
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = s
        metadata['topk_length'] = l
        metadata['topk_sequence'] = p
        metadata['length'] = [seq_len[0] for seq_len in l]
        metadata['sequence'] = [seq[:,0] for seq in p] 
        return decoder_outputs, decoder_hidden, metadata

    def _backtrack(self, nw_output, nw_hidden, predecessors, symbols, scores, b, hidden_size):
        """Backtracks over batch to generate optimal k-sequences.
        Args:
            nw_output [(batch*k, vocab_size)] * sequence_length: A Tensor of outputs from network
            nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length: A Tensor of hidden states from network
            predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
            symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
            scores [(batch*k)] * sequence_length: A Tensor containing sequence scores for every token t = [0, ... , seq_len - 1]
            b: Size of the batch
            hidden_size: Size of the hidden state
        Returns:
            output [(batch, k, vocab_size)] * sequence_length: A list of the output probabilities (p_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_t [(batch, k, hidden_size)] * sequence_length: A list containing the output features (h_n)
            from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
            h_n(batch, k, hidden_size): A Tensor containing the last hidden state for all top-k sequences.
            score [batch, k]: A list containing the final scores for all top-k sequences
            length [batch, k]: A list specifying the length of each sequence in the top-k candidates
            p (batch, k, sequence_len): A Tensor containing predicted sequence
        """

        lstm = isinstance(nw_hidden[0], tuple)

        # initialize return variables given different types
        output = list()
        h_t = list()
        p = list()
        # Placeholder for last hidden state of top-k sequences.
        # If a (top-k) sequence ends early in decoding, `h_n` contains
        # its hidden state when it sees EOS.  Otherwise, `h_n` contains
        # the last hidden state of decoding.
        if lstm:
            state_size = nw_hidden[0][0].size()
            if torch.cuda.is_available():
                h_n = tuple([torch.zeros(state_size).cuda(), torch.zeros(state_size).cuda()])
            else:
                h_n = tuple([torch.zeros(state_size), torch.zeros(state_size)])
        else:
            h_n = torch.zeros(nw_hidden[0].size())            
            if torch.cuda.is_available():
                h_n = h_n.cuda()

        l = [[self.rnn.max_length] * self.k for _ in range(b)]  # Placeholder for lengths of top-k sequences
                                                                # Similar to `h_n`

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(b, self.k).topk(self.k)
        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        batch_eos_found = [0] * b   # the number of EOS found
                                    # in the backward loop below for each batch

        t = self.rnn.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.

        t_predecessors = (sorted_idx + self.pos_index.view(b, self.k)).view(b * self.k)

        while t >= 0:
            #print(t_predecessors.view(b, self.k))            
            # Re-order the variables with the back pointer
            current_output = nw_output[t].index_select(0, t_predecessors)
            if lstm:
                current_hidden = tuple([h.index_select(1, t_predecessors) for h in nw_hidden[t]])
            else:
                current_hidden = nw_hidden[t].index_select(1, t_predecessors)
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of
            # the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors).squeeze()

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.squeeze(1).eq(self.EOS).nonzero()
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = int(idx[0] / self.k)
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_output[res_idx, :] = nw_output[t][idx[0], :]
                    if lstm:
                        current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]
                        current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]
                        h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data
                        h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data
                    else:
                        current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]
                        h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data
                    current_symbol[res_idx, :] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]].data[0]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            output.append(current_output)
            h_t.append(current_hidden)
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.data[0]] for k_idx in re_sorted_idx[b_idx,:]]

        re_sorted_idx = (re_sorted_idx + self.pos_index.view(b, self.k)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        output = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(output)]
        p = [step.index_select(0, re_sorted_idx).view(b, self.k, -1) for step in reversed(p)]
        if lstm:
            h_t = [tuple([h.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for h in step]) for step in reversed(h_t)]
            h_n = tuple([h.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size) for h in h_n])
        else:
            h_t = [step.index_select(1, re_sorted_idx).view(-1, b, self.k, hidden_size) for step in reversed(h_t)]
            h_n = h_n.index_select(1, re_sorted_idx.data).view(-1, b, self.k, hidden_size)
        s = s.data

        return output, h_t, h_n, s, l, p

