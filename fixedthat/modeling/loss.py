import numpy as np

import torch.nn as nn

from seq2seq.loss.loss import Loss

class BCELoss(Loss):
    """ Batch averaged poisson negative log-likelihood loss for predicting count data
        Args:
        log_input
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """
    
    _NAME = "Avg BCELoss"

    def __init__(self, weight=None, size_average=True, pad=None):
        self.size_average = size_average
        self.pad = pad

        super(BCELoss, self).__init__(
            self._NAME,
            nn.BCELoss(weight, size_average)) #, r))

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
        #print(outputs, target.float())
        if self.pad is not None:
            target = target.contiguous().view(-1)
            #print(outputs.shape, target.shape, target.ne(self.pad).shape)
            self.acc_loss += self.criterion(outputs.view(-1) * target.ne(self.pad).float(),
                                            target.float() * target.ne(self.pad).float())
        else:
            self.acc_loss += self.criterion(outputs, target.float())
        self.norm_term += 1
        
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
        #print(outputs, target.float())
        self.acc_loss += self.criterion(outputs, target.float())
        self.norm_term += 1

class MSELoss(Loss):
    """ Batch averaged poisson negative log-likelihood loss for predicting count data
        Args:
        log_input
        size_average (bool, optional): refer to http://pytorch.org/docs/master/nn.html#nllloss
    """
    
    _NAME = "Avg MSELoss"

    def __init__(self, size_average=True):
        self.size_average = size_average

        super(MSELoss, self).__init__(
            self._NAME,
            nn.MSELoss(size_average=size_average))

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
        #print(outputs, target)
        self.acc_loss += self.criterion(outputs, target.float())
        self.norm_term += 1

class MTLLoss(Loss):
    '''
    multi task learning loss for e.g. supervised attention
    
    '''

    _NAME = 'MTL Loss'

    def __init__(self, losses, lambdas=1):
        self.losses = losses
        self.lambdas = lambdas
        if type(lambdas) is int:
            self.lambdas = [1] * len(losses)
        super(MTLLoss, self).__init__(self._NAME, nn.modules.loss._Loss())

    def reset(self):
        for loss in self.losses:
            loss.reset()
        
    def backward(self):
        if any(type(loss.acc_loss) is int for loss in self.losses):
            raise ValueError("No loss to back propagate.")
        for index,loss in enumerate(self.losses[1:]):
            loss.acc_loss *= self.lambdas[index+1]
            loss.acc_loss.backward(retain_graph=True)
        self.losses[0].acc_loss *= self.lambdas[0]
        self.losses[0].backward()
        
    def cuda(self):
        for loss in self.losses:
            loss.criterion.cuda()
                    
    def get_loss(self):
        return np.array([loss.get_loss() for loss in self.losses])

    def eval_subloss_batch(self, i, outputs, target):
        self.losses[i].eval_batch(outputs, target)
        
    def eval_batch(self, outputs, target):
        self.losses[0].eval_batch(outputs, target)
