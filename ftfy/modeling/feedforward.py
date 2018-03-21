import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
use_cuda = torch.cuda.is_available()

class Feedforward(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, link=None):
        super(Feedforward, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.inp = nn.Linear(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.link = link

    #forward needs to take in the stack and input buffer (and output buffer? and maybe other state like counter?)
    def forward(self, input):

        input_hid = F.relu(self.inp(input))
        output = self.out(input_hid)
        if self.link is None:
            return output
        output = self.link(output)

