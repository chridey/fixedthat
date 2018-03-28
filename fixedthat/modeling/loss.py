import torch.nn as nn

from seq2seq.loss.loss import Loss

class PoissonLoss(Loss):
    """ Batch averaged poisson negative log-likelihood loss for predicting count data
        Args:
        log_input
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """
    
    _NAME = "Avg PoissonLoss"

    def __init__(self, log_input=True, size_average=True):
        self.size_average = size_average

        super(PoissonLoss, self).__init__(
            self._NAME,
            nn.PoissonNLLLoss(log_input=log_input, size_average=size_average))

    def get_loss(self):
        if isinstance(self.acc_loss, int):
            return 0
        # total loss for all batches
        loss = self.acc_loss.data[0]
        if self.size_average:
            # average loss per batch
            loss /= self.norm_term
        return loss

    def eval_batch(self, outputs, target):
        self.acc_loss += self.criterion(outputs, target.float())
        self.norm_term += 1
