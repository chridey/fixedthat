from __future__ import print_function, division

import os
import numpy as np

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss

class Evaluator(object):
    """ Class to evaluate models with given datasets.
    Args:
        loss (seq2seq.loss, optional): loss for evaluator (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for evaluator (default: 64)

        prediction_dir - whether to save individual predictions
    """

    def __init__(self, loss=NLLLoss(), batch_size=64, prediction_dir=None):
        self.loss = loss
        self.batch_size = batch_size

        self.prediction_dir = prediction_dir
        if not os.path.exists(prediction_dir):
            os.mkdir(prediction_dir)
        self.epoch = 0

    def save_predictions(self, predictions):
        #print(len(predictions), [i.shape for i in predictions])
        predictions = np.concatenate(predictions, axis=0)
        #print(predictions.shape)
        np.save(os.path.join(self.prediction_dir, 'predictions' + str(self.epoch)), predictions)
            
        self.epoch += 1
        
    def evaluate(self, model, data):
        """ Evaluate a model on given dataset and return performance.
        Args:
            model (seq2seq.models): model to evaluate
            data (seq2seq.dataset.dataset.Dataset): dataset to evaluate against
        Returns:
            loss (float): loss of the given model on the given dataset
        """
        model.eval()

        loss = self.loss
        loss.reset()
        match = 0
        total = 0

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=True, sort_key=lambda x: len(x.src),
            device=device, train=False)
        tgt_vocab = data.fields[seq2seq.tgt_field_name].vocab
        pad = tgt_vocab.stoi[data.fields[seq2seq.tgt_field_name].pad_token]

        predictions = []
        for batch in batch_iterator:
            input_variables, input_lengths  = getattr(batch, seq2seq.src_field_name)
            target_variables = getattr(batch, seq2seq.tgt_field_name)

            decoder_outputs, decoder_hidden, other = model(input_variables, input_lengths.tolist(), target_variables)

            # Evaluation
            seqlist = other['sequence']
            batch_predictions = []
            for step, step_output in enumerate(decoder_outputs):
                target = target_variables[:, step + 1]
                loss.eval_batch(step_output.view(target_variables.size(0), -1), target)

                non_padding = target.ne(pad)
                correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().data[0]
                match += correct
                total += non_padding.sum().data[0]

                if self.prediction_dir is not None:
                    batch_predictions.append(seqlist[step].view(-1,1).cpu().data.numpy())

            if self.prediction_dir is not None:
                predictions.append(np.concatenate(batch_predictions, axis=1))
                    
        if self.prediction_dir is not None:
            self.save_predictions(predictions)
                
        if total == 0:
            accuracy = float('nan')
        else:
            accuracy = match / total

        return loss.get_loss(), accuracy
