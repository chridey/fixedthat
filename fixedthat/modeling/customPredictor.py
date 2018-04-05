import torch
from torch.autograd import Variable

from seq2seq.evaluator import Predictor

class CustomPredictor(Predictor):
    def predict(self, src_seq, counter, verbose=True):
        src_id_seq = Variable(torch.LongTensor([self.src_vocab.stoi[tok] for tok in src_seq]),
                              volatile=True).view(1, -1)

        counts = Variable(torch.LongTensor([counter])).view(1, 1)
        
        if torch.cuda.is_available():
            src_id_seq = src_id_seq.cuda()
            counts = counts.cuda()
    
        softmax_list, _, other = self.model(src_id_seq, [len(src_seq)], counts=counts)

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

        return tgt_seq
