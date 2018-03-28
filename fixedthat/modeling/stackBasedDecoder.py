import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

class StackBasedDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size,
                 embedding=None, rnn=None):
        super(StackBasedDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        if embedding is None:
            self.embedding = nn.Embedding(output_size, hidden_size)
        else:
            self.embedding = embedding

        if rnn is None:
            self.gru = nn.GRU(hidden_size, hidden_size)
        else:
            self.gru = rnn

        self.empty_stack = Variable(torch.randn(input_size).type(torch.FloatTensor), requires_grad=True)
        self.empty_buffer = Variable(torch.randn(input_size).type(torch.FloatTensor), requires_grad=True)
        if use_cuda:
            self.empty_stack = self.empty_stack.cuda()
            self.empty_buffer = self.empty_buffer.cuda()
        
        self.inp = nn.Linear(3*input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    #forward needs to take in the stack and input buffer (and output buffer? and maybe other state like counter?)
    def forward(self, stack, input_buffer, output): #previous, hidden):

        #previous = self.embedding(previous).view(1, -1, self.input_size)
        #output, hidden = self.gru(previous, hidden)

        #TODO: add empty token
        if len(stack):
            top_stack = stack[-1][1]
        else:
            top_stack = self.empty_stack
            
        if len(input_buffer):
            front_buffer = input_buffer[0][1]
        else:
            front_buffer = self.empty_buffer
            
        #for now, concat with the top element of the stack and the buffer
        #now B x 3D
        input = torch.cat([top_stack.view(1,-1), front_buffer.view(1,-1), output[0]], dim=1)
                
        input_hid = F.relu(self.inp(input))
        output = self.softmax(self.out(input_hid))
        return output #, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
    
