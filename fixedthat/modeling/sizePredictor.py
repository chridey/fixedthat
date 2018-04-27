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

class SizePredictor(nn.Module):
    r"""
    Provides functionality for size prediction
    Args:
    Inputs: 
    Outputs: 
    """

    def __init__(self, input_size, hidden_size, bidirectional=False, use_attention=False, full=False):
        super(SizePredictor, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.bidirectional_encoder = bidirectional

        self.use_attention = use_attention
        if use_attention:
            self.attention = Attention(self.input_size)

        self.query = Variable(torch.randn(self.input_size).type(torch.FloatTensor), requires_grad=True)
        if torch.cuda.is_available():
            self.query = self.query.cuda()
        
        self.inp = nn.Linear(self.input_size, self.hidden_size)

        self.out = nn.Linear(self.hidden_size, 1, bias=False)
        
    def forward(self, encoder_outputs):
        batch_size = self._validate_args(encoder_outputs)
        
        query = self.query.view(1,1,-1).expand(batch_size, 1, -1)
        #print(query.shape)
        #print(encoder_outputs.shape)
            
        attn = None
        #fixed query attention
        if self.use_attention:
            output, attn = self.attention(query, encoder_outputs)
            output = output.view(batch_size, -1)
        else:
            output = torch.mean(encoder_outputs, dim=1)

        #print(output.shape)
        #prediction = torch.exp(self.out(self.inp(output)).view(-1))
        input_hid = F.relu(self.inp(output))
        #print(input_hid.shape)
        prediction = self.out(input_hid).view(-1)
        #print(prediction)
        #print(torch.exp(prediction))
        #print(torch.exp(prediction).long())
        #print(attn.shape)
        #print('bias', self.out.bias)
        return [prediction], None, {'attention_score': attn, 'sequence': [prediction.long()]} #[torch.exp(prediction).long()]}

    def _validate_args(self, encoder_outputs):
                    
        if encoder_outputs is None:
            raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        batch_size = encoder_outputs.size(0)
            
        return batch_size
                
