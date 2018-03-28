import torch
import torch.nn.functional as F
from torch.autograd import Variable

from seq2seq.models.TopKDecoder import TopKDecoder, _inflate

class BeamSearchDecoder(TopKDecoder):
    def forward(self, encoder_inputs, decoder_inputs=None, encoder_hidden=None, encoder_outputs=None,
                function=F.log_softmax, teacher_forcing_ratio=0, retain_output_probs=True, counts=None):
        """
        Forward rnn for MAX_LENGTH steps.  Look at :func:`seq2seq.models.DecoderRNN.DecoderRNN.forward_rnn` for details.
        """

        inputs, batch_size, max_length = self.rnn._validate_args(encoder_inputs, decoder_inputs, encoder_hidden,
                                                                 encoder_outputs, function, teacher_forcing_ratio,
                                                                 counts)

        self.pos_index = Variable(torch.LongTensor(range(batch_size)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: b*k x h
        encoder_hidden = self.rnn._init_state(encoder_hidden)
        if encoder_hidden is None:
            hidden = None
        else:
            if isinstance(encoder_hidden, tuple):
                hidden = tuple([_inflate(h, self.k, 1) for h in encoder_hidden])
            else:
                hidden = _inflate(encoder_hidden, self.k, 1)
                
        # ... same idea for encoder_outputs and decoder_outputs
        if self.rnn.use_attention:
            inflated_encoder_outputs = _inflate(encoder_outputs, self.k, 0)
        else:
            inflated_encoder_outputs = None

        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = torch.Tensor(batch_size * self.k, 1)
        sequence_scores.fill_(-float('Inf'))
        sequence_scores.index_fill_(0, torch.LongTensor([i * self.k for i in range(0, batch_size)]), 0.0)
        sequence_scores = Variable(sequence_scores)

        # Initialize the input vector
        #input_var = Variable(torch.transpose(torch.LongTensor([[self.SOS] * batch_size * self.k]), 0, 1))
        #inputs is B x 1 x C (1 or 2) -> B x K x C -> B * K x C
        input_var = inputs.view(batch_size, 1, -1).expand(batch_size, self.k, -1).view(batch_size*self.k, -1)
        
        # Store decisions for backtracking
        stored_outputs = list()
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        stored_hidden = list()

        for _ in range(0, max_length):
            # Run the RNN one step forward
            log_softmax_output, hidden, _ = self.rnn.forward_step(input_var, hidden,
                                                                  inflated_encoder_outputs, function=function)

            # If doing local backprop (e.g. supervised training), retain the output layer
            if retain_output_probs:
                stored_outputs.append(log_softmax_output)

            # To get the full sequence scores for the new candidates, add the local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = _inflate(sequence_scores, self.V, 1)
            sequence_scores += log_softmax_output.squeeze(1)

            #need to change the decode function to handle legal inputs instead of the code below
            scores, candidates, input_var, predecessors = self.decode()            
            ####
            '''
            scores, candidates = sequence_scores.view(batch_size, -1).topk(self.k, dim=1)

            # Reshape input = (bk, 1) and sequence_scores = (bk, 1)
            input_var = (candidates % self.V).view(batch_size * self.k, 1)
            sequence_scores = scores.view(batch_size * self.k, 1)

            predecessors = (candidates / self.V + self.pos_index.expand_as(candidates)).view(batch_size * self.k, 1)            '''
            ####
            
            # Update fields for next timestep
            if isinstance(hidden, tuple):
                hidden = tuple([h.index_select(1, predecessors.squeeze()) for h in hidden])
            else:
                hidden = hidden.index_select(1, predecessors.squeeze())

            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())
            eos_indices = input_var.data.eq(self.EOS)
            if eos_indices.nonzero().dim() > 0:
                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
            stored_hidden.append(hidden)

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
        metadata['inputs'] = inputs
        metadata['output'] = output
        metadata['h_t'] = h_t
        metadata['score'] = s
        metadata['topk_length'] = l
        metadata['topk_sequence'] = p
        metadata['length'] = [seq_len[0] for seq_len in l]
        metadata['sequence'] = [seq[0] for seq in p]
        return decoder_outputs, decoder_hidden, metadata
            
