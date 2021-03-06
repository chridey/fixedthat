'''
the model for predicting which tokens to copy to the output, as described in Section 4.3 of "Fixed That for You: Generating Contrastive Claims with Semantic Edits"
'''

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

from seq2seq.models.attention import Attention    
from seq2seq.models import DecoderRNN

class CopyPredictor(nn.Module):
    r"""
    Provides functionality for binary copy prediction
    """

    def __init__(self, input_size, hidden_size, bidirectional=False, use_attention=False, beta=0, gamma=(1-10e-5), crf=False):
        super(CopyPredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional_encoder = bidirectional
        self.beta = beta
        self.old_beta = None
        
        self.batches_seen = 0
        self.gamma = gamma
        
        self.use_attention = False #use_attention
        if self.use_attention:
            self.attention = Attention(self.input_size)

            self.query = Variable(torch.randn(self.input_size).type(torch.FloatTensor), requires_grad=True)
            if torch.cuda.is_available():
                self.query = self.query.cuda()
        
        self.inp = nn.Linear(self.input_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, 1, bias=False)

    def reset(self):
        assert(self.old_beta is not None)
        self.beta = self.old_beta
        self.old_beta = None

    def anneal(self):
        if self.batches_seen == 0:
            self.beta = self.gamma
        else:
            self.beta *= self.gamma
        self.batches_seen += 1

    def use_gold(self, use_gold):
        self.old_beta = self.beta
        #for toggling dev/test mode where when we always want the predictions not gold
        if use_gold:
            self.beta = 1
        else:
            self.beta = 0
    
    def _get_logits(self, output, batch_size, input_size):
        input_hid = F.relu(self.inp(output.contiguous().view(batch_size*input_size, -1)))
        prediction = F.sigmoid(self.out(input_hid).view(batch_size, input_size)).transpose(0,1).contiguous()
        return prediction

    def forward(self, encoder_outputs, mask=None, input_lengths=None):
        
        if mask is not None and self.beta==1:
            return mask.transpose(0,1).contiguous().float(), None, None
        
        batch_size, input_size = self._validate_args(encoder_outputs)

        attn = None        
        if self.use_attention:
            query = self.query.view(1,1,-1).expand(batch_size, 1, -1)
            
            #fixed query attention
            output, attn = self.attention(query, encoder_outputs)
            output = output.view(batch_size, -1)
        else:
            output = encoder_outputs
            
        prediction = self._get_logits(output, batch_size, input_size)

        #for interpolating between gold and predicted during training
        scores = self.beta*mask.transpose(0,1).float() + (1-self.beta)*prediction
        predictions = (scores+0.5).long()
        
        return scores, None, {'attention_score': attn, 'sequence': predictions}

    def score(self, logits, y, lens):
        y_exp = y.unsqueeze(-1)
        scores = torch.gather(logits, 2, y_exp).squeeze(-1)
        mask = sequence_mask(lens).float()
        scores = scores * mask
        score = scores.sum(1).squeeze(-1)

        return score

    def _validate_args(self, encoder_outputs):
                    
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)
        input_size = encoder_outputs.size(1)        
            
        return batch_size, input_size
                
