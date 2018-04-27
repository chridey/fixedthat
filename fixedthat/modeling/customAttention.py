import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    .. math::
            \begin{array}{ll}
            x = context*output \\
            attn = exp(x_i) / sum_j exp(x_j) \\
            output = \tanh(w * (attn * context) + b * output)
            \end{array}
    Args:
        dim(int): The number of expected features in the output
    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.
    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.
    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.
    Examples::
         >>> attention = seq2seq.models.Attention(256)
         >>> context = Variable(torch.randn(5, 3, 256))
         >>> output = Variable(torch.randn(5, 5, 256))
         >>> output, attn = attention(output, context)
    """
    def __init__(self, dim, copy_predictor=False):
        super(Attention, self).__init__()
        self.multiplier = 2
        if copy_predictor:
            self.multiplier = 3
            
        self.linear_out = nn.Linear(dim*self.multiplier, dim)
                        
        self.mask = None

        self.copy_predictor = copy_predictor

    def set_mask(self, mask):
        """
        Sets indices to be masked
        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context, mask=None, copy_mask=None):        
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        '''
        if self.copy_predictor:
            assert(mask is not None)
            mask_e = mask.unsqueeze(2).expand_as(context)
            copy_context = mask_e.float() * context * mask_e.ne(2).float()
            context = (1-mask_e).float() * context * mask_e.ne(2).float()
        '''
        
        # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)        
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            #attn.data.masked_fill_(mask, -float('inf'))
            attn.data.masked_fill_(mask.data.ne(2), -float('inf'))
        attn = F.softmax(attn.view(-1, input_size)).view(batch_size, -1, input_size)

        # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)

        if self.copy_predictor:
            # (batch, out_len, dim) * (batch, in_len, dim) -> (batch, out_len, in_len)
            #TODO: separate attention
            #copy_attn = torch.bmm(output, copy_context.transpose(1, 2))
            #if self.mask is not None:
            #    copy_attn.data.masked_fill_(self.mask, -float('inf'))
            #copy_attn = F.softmax(copy_attn.view(-1, input_size)).view(batch_size, -1, input_size)

            # (batch, out_len, in_len) * (batch, in_len, dim) -> (batch, out_len, dim)
            mask_e = copy_mask.view(batch_size, -1, input_size).expand_as(attn)
            sub_attn = (1-mask_e).float() * attn * mask.view(batch_size, -1, input_size).expand_as(attn).ne(2).float()
            copy_attn = mask_e.float() * attn * mask.view(batch_size, -1, input_size).expand_as(attn).ne(2).float()
            
            sub_mix = torch.bmm(sub_attn, context)            
            copy_mix = torch.bmm(copy_attn, context)            
            
            combined = torch.cat((sub_mix, copy_mix, output), dim=2)
            attn = torch.cat((copy_mask.view(batch_size, 1, input_size).expand_as(attn).unsqueeze(3).float(),
                              attn.unsqueeze(3), copy_attn.unsqueeze(3)), dim=3)
        else:
            mix = torch.bmm(attn, context)
            # concat -> (batch, out_len, 2*dim)
            combined = torch.cat((mix, output), dim=2)
            # output -> (batch, out_len, dim)
            
        output = F.tanh(self.linear_out(combined.view(-1, self.multiplier * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn
