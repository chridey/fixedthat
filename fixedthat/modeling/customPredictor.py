from __future__ import print_function

import numpy as np

import torch
from torch.autograd import Variable

import torchtext

import seq2seq
from seq2seq.evaluator import Predictor

class CustomPredictor(Predictor):

    def __init__(self, model, src_vocab=None, tgt_vocab=None):
        """
        Predictor class to evaluate for a given model.
        Args:
            model (seq2seq.models): trained model. This can be loaded from a checkpoint
                using `seq2seq.util.checkpoint.load`
            src_vocab (seq2seq.dataset.vocabulary.Vocabulary): source sequence vocabulary
            tgt_vocab (seq2seq.dataset.vocabulary.Vocabulary): target sequence vocabulary
        """
        if torch.cuda.is_available():
            self.model = model.cuda()
        else:
            self.model = model.cpu()
        self.model.eval()
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab

        self.token_to_string = True
        if src_vocab is None or tgt_vocab is None:
            self.token_to_string = False
    
    def predict_batch(self, data, batch_size, file_handle, use_counter=False):
        
        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)

        #pad = self.tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        if use_counter:
            counter = 1
            counts = Variable(torch.LongTensor([counter] * batch_size)).view(batch_size, 1)

            if torch.cuda.is_available():
                counts = counts.cuda()
        
        output = []
        idxs = []
        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            batch_idxs = getattr(batch, 'idx')
            copy_mask = getattr(batch, 'extra', None)

            if copy_mask is not None:
                self.model.copy_predictor.use_gold(False)
            
            if use_counter:
                _, _, other = self.model(input_variables, input_lengths.tolist(),
                                        counts=counts[:input_variables.size(0),:],
                                        mask=copy_mask)
            else:
                _, _, other = self.model(input_variables, input_lengths.tolist(),
                                         mask=copy_mask)
                    
            seqlist = other['sequence']
            #print('seq', seqlist)

            batch_output = [[batch_idxs[i]] for i in range(input_variables.size(0))]
            for step, token_ids in enumerate(seqlist):
                for index, token_id in enumerate(token_ids):
                    if not self.token_to_string or step < other['length'][index]-1:
                        if self.token_to_string:
                            batch_output[index].append(self.tgt_vocab.itos[int(token_id.data)])
                        else:
                            batch_output[index].append(int(token_id.data))

            for seq in batch_output:
                print(' '.join(map(unicode, seq)).encode('utf-8'), file=file_handle)
            #output.extend(batch_output)

        return output
                    
    def predict(self, src_seq, counter=None, start=0, end=5, count_normalize=False, verbose=False):
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)

        if counter is not None:
            counter_range = [counter]
        else:
            counter_range = list(range(start, end+1))

        best = []
        best_scores = []
        for counter in counter_range:
        
            counts = Variable(torch.LongTensor([counter])).view(1, 1)

            if torch.cuda.is_available():
                src_id_seq = src_id_seq.cuda()
                counts = counts.cuda()

            softmax_list, _, other = self.model(src_id_seq, [len(src_seq)], counts=counts)

            #if we are in beam search mode
            if verbose and 'score' in other:
                #print(other['score'].shape)
                #print(other['score'])
                #print(other['topk_sequence'])
                #print(other['topk_length'])

                for i in range(other['score'].size(1)):
                    length = other['topk_length'][0][i]

                    tgt_id_seq = [other['topk_sequence'][di][0][i].data[0] for di in range(length)]
                    tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
                    print(tgt_seq, other['score'][0][i])

            length = other['length'][0]

            tgt_id_seq = [other['sequence'][di][0].data[0] for di in range(length)]
            tgt_seq = [self.tgt_vocab.itos[tok] for tok in tgt_id_seq]
    
            best.append(tgt_seq)
            #note: requires beam search mode
            best_scores.append(other['score'][0][0])

            if count_normalize:
                best_scores[-1] /= counter

        print(best, best_scores)
        return best[np.argmax(best_scores)]
            
        #return tgt_seq
