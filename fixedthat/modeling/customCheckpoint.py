'''
adapted from https://github.com/IBM/pytorch-seq2seq to allow saving and loading of additional vocabularies (e.g. for topics/subreddits)
'''

from __future__ import print_function
import os
import time
import shutil

import torch
import dill

from seq2seq.util.checkpoint import Checkpoint

class CustomCheckpoint(Checkpoint):
    """
    The Checkpoint class manages the saving and loading of a model during training. It allows training to be suspended
    and resumed at a later time (e.g. when running on a cluster using sequential jobs).
    To make a checkpoint, initialize a Checkpoint object with the following args; then call that object's save() method
    to write parameters to disk.
    Args:
        model (seq2seq): seq2seq model being trained
        optimizer (Optimizer): stores the state of the optimizer
        epoch (int): current epoch (an epoch is a loop through the full training data)
        step (int): number of examples seen within the current epoch
        input_vocab (Vocabulary): vocabulary for the input language
        output_vocab (Vocabulary): vocabulary for the output language
        extra_vocabs: list of Vocabularies
    Attributes:
        CHECKPOINT_DIR_NAME (str): name of the checkpoint directory
        TRAINER_STATE_NAME (str): name of the file storing trainer states
        MODEL_NAME (str): name of the file storing model
        INPUT_VOCAB_FILE (str): name of the input vocab file
        OUTPUT_VOCAB_FILE (str): name of the output vocab file
        EXTRA_VOCAB_FILE_PREFIX (str): beginning of extra vocab file
    """

    CHECKPOINT_DIR_NAME = 'checkpoints'
    TRAINER_STATE_NAME = 'trainer_states.pt'
    MODEL_NAME = 'model.pt'
    INPUT_VOCAB_FILE = 'input_vocab.pt'
    OUTPUT_VOCAB_FILE = 'output_vocab.pt'

    EXTRA_VOCAB_FILE_PREFIX = 'extra_vocab'

    def __init__(self, model, optimizer, epoch, step, input_vocab, output_vocab, path=None, extra_vocabs=None):
        self.model = model
        self.optimizer = optimizer
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab
        self.epoch = epoch
        self.step = step
        self._path = path

        self.extra_vocabs = extra_vocabs

    def save(self, experiment_dir):
        """
        Saves the current model and related training parameters into a subdirectory of the checkpoint directory.
        The name of the subdirectory is the current local time in Y_M_D_H_M_S format.
        Args:
            experiment_dir (str): path to the experiment root directory
        Returns:
             str: path to the saved checkpoint subdirectory
        """
        date_time = time.strftime('%Y_%m_%d_%H_%M_%S', time.localtime())

        self._path = os.path.join(experiment_dir, self.CHECKPOINT_DIR_NAME, date_time)
        path = self._path

        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        torch.save({'epoch': self.epoch,
                    'step': self.step,
                    'optimizer': self.optimizer
                   },
                   os.path.join(path, self.TRAINER_STATE_NAME))
        torch.save(self.model, os.path.join(path, self.MODEL_NAME))

        with open(os.path.join(path, self.INPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.input_vocab, fout)
        with open(os.path.join(path, self.OUTPUT_VOCAB_FILE), 'wb') as fout:
            dill.dump(self.output_vocab, fout)

        if self.extra_vocabs is not None:
            for idx, extra_vocab in enumerate(self.extra_vocabs):
                extra_vocab_file = self.EXTRA_VOCAB_FILE_PREFIX + str(idx) + '.pt'
                with open(os.path.join(path, extra_vocab_file), 'wb') as fout:
                    dill.dump(extra_vocab, fout)
            
        return path

    @classmethod
    def load(cls, path):
        """
        Loads a Checkpoint object that was previously saved to disk.
        Args:
            path (str): path to the checkpoint subdirectory
        Returns:
            checkpoint (Checkpoint): checkpoint object with fields copied from those stored on disk
        """
        if torch.cuda.is_available():
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME))
            model = torch.load(os.path.join(path, cls.MODEL_NAME))
        else:
            resume_checkpoint = torch.load(os.path.join(path, cls.TRAINER_STATE_NAME), map_location=lambda storage, loc: storage)
            model = torch.load(os.path.join(path, cls.MODEL_NAME), map_location=lambda storage, loc: storage)

        model.flatten_parameters() # make RNN parameters contiguous
        with open(os.path.join(path, cls.INPUT_VOCAB_FILE), 'rb') as fin:
            input_vocab = dill.load(fin)
        with open(os.path.join(path, cls.OUTPUT_VOCAB_FILE), 'rb') as fin:
            output_vocab = dill.load(fin)
        optimizer = resume_checkpoint['optimizer']

        extra_vocabs = []
        idx = 0
        extra_vocab_file = os.path.join(path, cls.EXTRA_VOCAB_FILE_PREFIX + str(idx) + '.pt')
        while os.path.exists(extra_vocab_file):
            with open(extra_vocab_file) as f:
                extra_vocabs.append(dill.load(f))
            idx += 1
            extra_vocab_file = os.path.join(path, cls.EXTRA_VOCAB_FILE_PREFIX + str(idx) + '.pt')

        return CustomCheckpoint(model=model, input_vocab=input_vocab,
                          output_vocab=output_vocab,
                          optimizer=optimizer,
                          epoch=resume_checkpoint['epoch'],
                          step=resume_checkpoint['step'],
                          path=path,
                          extra_vocabs=extra_vocabs)

