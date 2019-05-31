'''
adapted from https://github.com/IBM/pytorch-seq2seq to allow for the additional features described in Section 4 of "Fixed That for You: Generating Contrastive Claims with Semantic Edits"
'''

from torch import nn
import torch.nn.functional as F

from seq2seq.models import Seq2seq

class CustomSeq2seq(Seq2seq):
    '''
    encoder - the model for representing the source
    decoder - the model for predicting the target
    copy_predictor - the model to predict whether to copy from the input (described in Section 4.3)
    crf - use a CRF instead of independent binary copy predictions
    '''
    def __init__(self, encoder, decoder, decode_function=F.log_softmax, predict_size=False, 
                 copy_predictor=None, crf=False):

        super(CustomSeq2seq, self).__init__(encoder, decoder, decode_function=F.log_softmax)
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.predict_size = predict_size
        self.copy_predictor = copy_predictor
        
        if crf:
            self.crf = CRF(2)
        else:
            self.crf = None

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        if getattr(self.decoder, 'rnn', None) is not None:
            self.decoder.rnn.flatten_parameters()        
        
    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, counts=None, mask=None, filter_illegal=True, use_prefix=False,
                features=None, train_mode=True, cce_keywords=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        if self.predict_size:
            result = self.decoder(encoder_outputs)
        else:
            if self.copy_predictor is not None:
                copy_mask, _, other = self.copy_predictor(encoder_outputs, mask)
            else:
                copy_mask = mask
            result = self.decoder(input_variable,
                                decoder_inputs=target_variable,
                                encoder_hidden=encoder_hidden,
                                encoder_outputs=encoder_outputs,
                                function=self.decode_function,
                                teacher_forcing_ratio=teacher_forcing_ratio,
                                counts=counts,
                                mask=mask,
                                copy_mask=copy_mask,
                                filter_illegal=filter_illegal,
                                use_prefix=use_prefix,
                                  features=features,
                                  cce_keywords=cce_keywords)
            if self.copy_predictor is not None:
                result[-1]['copy_mask'] = copy_mask
                result[-1]['has_grad'] = self.copy_predictor.beta < 1

        transition_score = None
        if getattr(self, 'crf', None) is not None:
            if train_mode:
                labels = mask
                if self.predict_size:
                    labels = target_variable
                result[-1]['crf_score'] = self.score(encoder_outputs, input_lengths, labels)
            else:
                if self.predict_size:
                    scores = self.decoder._get_logits(output, batch_size, input_size)
                else:
                    scores = self.copy_predictor._get_logits(output, batch_size, input_size)
                scores, predictions = self.crf.viterbi_decode(scores, input_lengths)
                result[-1]['predictions'] = predictions
                result[-1]['crf_score'] = scores

        return result
                
    def score(self, encoder_outputs, input_lengths, labels):
        transition_score = self.crf.transition_score(labels, input_lengths)

        batch_size, input_size = encoder_outputs.size(0), encoder_outputs.size(1)
        if self.predict_size:
            scores = self.decoder._get_logits(output, batch_size, input_size)
            bilstm_score = self.decoder.score(scores, labels, input_lengths)
        else:
            scores = self.copy_predictor._get_logits(output, batch_size, input_size)
            bilstm_score = self.copy_predictor.score(scores, labels, input_lengths)

        norm_score = self.crf(scores, input_lengths)

        return transition_score + bilstm_score - norm_score

import torch.nn.init as I

def log_sum_exp(vec, dim=0):
    max, idx = torch.max(vec, dim)
    max_exp = max.unsqueeze(-1).expand_as(vec)
    return max + torch.log(torch.sum(torch.exp(vec - max_exp), dim))

class CRF(nn.Module):
    def __init__(self, vocab_size):
        super(CRF, self).__init__()

        self.vocab_size = vocab_size
        self.n_labels = n_labels = vocab_size + 2
        self.start_idx = n_labels - 2
        self.stop_idx = n_labels - 1
        self.transitions = nn.Parameter(torch.randn(n_labels, n_labels))

    def reset_parameters(self):
        I.normal(self.transitions.data, 0, 1)

    def forward(self, logits, lens):
        """
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        alpha = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        alpha[:, self.start_idx] = 0
        alpha = Variable(alpha)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        for logit in logits_t:
            logit_exp = logit.unsqueeze(-1).expand(batch_size,
                                                   *self.transitions.size())
            alpha_exp = alpha.unsqueeze(1).expand(batch_size,
                                                  *self.transitions.size())
            trans_exp = self.transitions.unsqueeze(0).expand_as(alpha_exp)
            mat = trans_exp + alpha_exp + logit_exp
            alpha_nxt = log_sum_exp(mat, 2).squeeze(-1)

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(alpha)
            alpha = mask * alpha_nxt + (1 - mask) * alpha
            c_lens = c_lens - 1

        alpha = alpha + self.transitions[self.stop_idx].unsqueeze(0).expand_as(alpha)
        norm = log_sum_exp(alpha, 1).squeeze(-1)

        return norm

    def viterbi_decode(self, logits, lens):
        """Borrowed from pytorch tutorial
        Arguments:
            logits: [batch_size, seq_len, n_labels] FloatTensor
            lens: [batch_size] LongTensor
        """
        batch_size, seq_len, n_labels = logits.size()
        vit = logits.data.new(batch_size, self.n_labels).fill_(-10000)
        vit[:, self.start_idx] = 0
        vit = Variable(vit)
        c_lens = lens.clone()

        logits_t = logits.transpose(1, 0)
        pointers = []
        for logit in logits_t:
            vit_exp = vit.unsqueeze(1).expand(batch_size, n_labels, n_labels)
            trn_exp = self.transitions.unsqueeze(0).expand_as(vit_exp)
            vit_trn_sum = vit_exp + trn_exp
            vt_max, vt_argmax = vit_trn_sum.max(2)

            vt_max = vt_max.squeeze(-1)
            vit_nxt = vt_max + logit
            pointers.append(vt_argmax.squeeze(-1).unsqueeze(0))

            mask = (c_lens > 0).float().unsqueeze(-1).expand_as(vit_nxt)
            vit = mask * vit_nxt + (1 - mask) * vit

            mask = (c_lens == 1).float().unsqueeze(-1).expand_as(vit_nxt)
            vit += mask * self.transitions[ self.stop_idx ].unsqueeze(0).expand_as(vit_nxt)

            c_lens = c_lens - 1

        pointers = torch.cat(pointers)
        scores, idx = vit.max(1)
        idx = idx.squeeze(-1)
        paths = [idx.unsqueeze(1)]

        for argmax in reversed(pointers):
            idx_exp = idx.unsqueeze(-1)
            idx = torch.gather(argmax, 1, idx_exp)
            idx = idx.squeeze(-1)

            paths.insert(0, idx.unsqueeze(1))

        paths = torch.cat(paths[1:], 1)
        scores = scores.squeeze(-1)

        return scores, paths

    def transition_score(self, labels, lens):
        """
        Arguments:
             labels: [batch_size, seq_len] LongTensor
             lens: [batch_size] LongTensor
        """
        batch_size, seq_len = labels.size()

        # pad labels with <start> and <stop> indices
        labels_ext = Variable(labels.data.new(batch_size, seq_len + 2))
        labels_ext[:, 0] = self.start_idx
        labels_ext[:, 1:-1] = labels
        mask = sequence_mask(lens + 1, max_len=seq_len + 2).long()
        pad_stop = Variable(labels.data.new(1).fill_(self.stop_idx))
        pad_stop = pad_stop.unsqueeze(-1).expand(batch_size, seq_len + 2)
        labels_ext = (1 - mask) * pad_stop + mask * labels_ext
        labels = labels_ext

        trn = self.transitions

        # obtain transition vector for each label in batch and timestep
        # (except the last ones)
        trn_exp = trn.unsqueeze(0).expand(batch_size, *trn.size())
        lbl_r = labels[:, 1:]
        lbl_rexp = lbl_r.unsqueeze(-1).expand(*(lbl_r.size() + [trn.size(0)]))
        trn_row = torch.gather(trn_exp, 1, lbl_rexp)

        # obtain transition score from the transition vector for each label
        # in batch and timestep (except the first ones)
        lbl_lexp = labels[:, :-1].unsqueeze(-1)
        trn_scr = torch.gather(trn_row, 2, lbl_lexp)
        trn_scr = trn_scr.squeeze(-1)

        mask = sequence_mask(lens + 1).float()
        trn_scr = trn_scr * mask
        score = trn_scr.sum(1).squeeze(-1)

        return score

def sequence_mask(lens, max_len=None):
    batch_size = lens.size(0)

    if max_len is None:
        max_len = lens.max().data[0]

    ranges = torch.arange(0, max_len).long()
    ranges = ranges.unsqueeze(0).expand(batch_size, max_len)
    ranges = Variable(ranges)

    if lens.data.is_cuda:
        ranges = ranges.cuda()

    lens_exp = lens.unsqueeze(1).expand_as(ranges)
    mask = ranges < lens_exp

    return mask
            
