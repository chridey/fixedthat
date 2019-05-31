import logging

import torch
import torchtext

class CustomTargetField(torchtext.data.Field):
    """ Wrapper class of torchtext.data.Field that forces batch_first to be True and prepend <sos> and append <eos> to sequences in preprocessing step.

    Attributes:
        sos_id: index of the start of sentence symbol
        eos_id: index of the end of sentence symbol
    """

    SYM_SOS = '<sos>'
    SYM_EOS = '<eos>'

    def __init__(self, **kwargs):
        logger = logging.getLogger(__name__)

        if kwargs.get('batch_first') == False:
            logger.warning("Option batch_first has to be set to use pytorch-seq2seq.  Changed to True.")
        kwargs['batch_first'] = True
        
        start = [self.SYM_SOS]
        if kwargs.get('add_start') is not None:
            if kwargs.pop('add_start') == False:
                start = []

        end = [self.SYM_EOS]
        def preprocess(start, seq, end):
            return start + seq + end

        self.get_size = False
        if kwargs.get('get_size') is not None:
            self.get_size = kwargs.pop('get_size')
            if self.get_size:
                start = []
                end = []
                        
        self.counter = False        
        if kwargs.get('counter') is not None:
            if kwargs.pop('counter') == True:
                def preprocess(start, seq, end):
                    seq, counts = seq[:len(seq) // 2], map(int, seq[len(seq) // 2:])
                    end = [(end[0], 0)]
                    if len(start):
                        start = [(start[0], counts[0])]
                    seq = zip(seq, counts[1:])

                    return start + seq + end
                
                self.counter = True

        if kwargs.get('preprocessing') is None:
            kwargs['preprocessing'] = lambda seq: preprocess(start, seq, end)
        else:
            func = kwargs['preprocessing']
            kwargs['preprocessing'] = lambda seq: preprocess(start, func(seq), end)

        self.sos_id = None
        self.eos_id = None
        super(CustomTargetField, self).__init__(**kwargs)

    def build_vocab(self, *args, **kwargs):
        super(CustomTargetField, self).build_vocab(*args, **kwargs)
        self.sos_id = self.vocab.stoi[self.SYM_SOS]
        self.eos_id = self.vocab.stoi[self.SYM_EOS]

    def process(self, batch, device, train):
        if self.get_size:
            size = torch.autograd.Variable(torch.LongTensor([int(x[0]) for x in batch])).view(-1, 1).expand(-1, 2)
            if torch.cuda.is_available():
                size = size.cuda()
            return size
        
        if not self.counter:
            return super(CustomTargetField, self).process(batch, device, train)

        batch = list(batch)
        batch_tokens = []
        batch_counts = []
        for x in batch:
            tokens, counts = zip(*x)
            new_counts = []
            decrement = 0
            #need to handle UNK tokens by decrementing counts
            for i in range(1,len(tokens))[::-1]:
                new_counts.insert(0, counts[i] - decrement)
                if counts[i] != counts[i-1] and self.vocab.stoi[tokens[i]] == 0:
                    decrement += 1
            new_counts.insert(0, counts[0] - decrement)
            
            batch_tokens.append(tokens)
            batch_counts.append(counts)

        padded = self.pad(batch_tokens)
        tensor = self.numericalize(padded, device=device, train=train)

        save_pad = self.pad_token
        save_vocab = self.use_vocab
        save_lengths = self.include_lengths
        self.pad_token = 0
        self.use_vocab = False
        self.include_lengths = False

        padded = self.pad(batch_counts)
        count_tensor = self.numericalize(padded, device=device, train=train)

        self.pad_token = save_pad
        self.use_vocab = save_vocab
        self.include_lengths = save_lengths
        if self.include_lengths:
            tensor, lengths = tensor

        tensor = torch.cat([tensor.unsqueeze(2), count_tensor.unsqueeze(2)], dim=2)
        
        if self.include_lengths:
            return tensor, lengths
        
        return tensor
