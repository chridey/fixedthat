import os
import logging

import torch

from seq2seq.loss import NLLLoss
from seq2seq.trainer import SupervisedTrainer
from seq2seq.evaluator import Evaluator

class CustomTrainer(SupervisedTrainer):
    """ The SupervisedTrainer class helps in setting up a training framework in a
    supervised setting.
    Args:
        expt_dir (optional, str): experiment Directory to store details of the experiment,
            by default it makes a folder in the current directory to store the details (default: `experiment`).
        loss (seq2seq.loss.loss.Loss, optional): loss for training, (default: seq2seq.loss.NLLLoss)
        batch_size (int, optional): batch size for experiment, (default: 64)
        checkpoint_every (int, optional): number of epochs to checkpoint after, (default: 100)
    """
    def __init__(self, expt_dir='experiment', loss=NLLLoss(), batch_size=64,
                 random_seed=None,
                 checkpoint_every=100, print_every=100,
                 evaluator=None):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss

        self.evaluator = evaluator
        if self.evaluator is None:
            self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size)
            
        self.optimizer = None
        self.checkpoint_every = checkpoint_every
        self.print_every = print_every

        if not os.path.isabs(expt_dir):
            expt_dir = os.path.join(os.getcwd(), expt_dir)
        self.expt_dir = expt_dir
        if not os.path.exists(self.expt_dir):
            os.makedirs(self.expt_dir)
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
