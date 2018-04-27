import os
import logging

import numpy as np

import torch
import torchtext

import seq2seq
from seq2seq.loss import NLLLoss
from seq2seq.trainer import SupervisedTrainer
from seq2seq.evaluator import Evaluator
from seq2seq.util.checkpoint import Checkpoint

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
                 evaluator=None, filter_illegal=True, use_prefix=False):
        self._trainer = "Simple Trainer"
        self.random_seed = random_seed
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
        self.loss = loss

        self.evaluator = evaluator
        if self.evaluator is None:
            self.evaluator = Evaluator(loss=self.loss, batch_size=batch_size, filter_illegal=filter_illegal,
                                       use_prefix=use_prefix)
        self.filter_illegal = filter_illegal
        self.use_prefix = use_prefix
        
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

    def _train_batch(self, input_variable, input_lengths, target_variable, model, teacher_forcing_ratio,
                     copy_mask=None):
        loss = self.loss
        # Forward propagation
        decoder_outputs, decoder_hidden, other = model(input_variable, input_lengths, target_variable,
                                                       teacher_forcing_ratio=teacher_forcing_ratio,
                                                       mask=copy_mask, filter_illegal=self.filter_illegal,
                                                       use_prefix=self.use_prefix)
        # Get loss
        loss.reset()

        if 'copy_mask' in other:
            for step, step_output in enumerate(other['copy_mask']):
                batch_size = target_variable.size(0)
                #print(target_variable.shape, copy_mask.shape, step_output.shape, other['copy_mask'].shape)
                loss.eval_subloss_batch(1, step_output.contiguous().view(batch_size, -1),
                                        copy_mask[:, step])
                
        for step, step_output in enumerate(decoder_outputs):
            batch_size = target_variable.size(0)
            loss.eval_batch(step_output.contiguous().view(batch_size, -1), target_variable[:, step + 1])
        # Backward propagation
        model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.get_loss()

    def _train_epoches(self, data, model, n_epochs, start_epoch, start_step,
                       dev_data=None, teacher_forcing_ratio=0):
        log = self.logger

        print_loss_total = 0  # Reset every print_every
        epoch_loss_total = 0  # Reset every epoch

        device = None if torch.cuda.is_available() else -1
        batch_iterator = torchtext.data.BucketIterator(
            dataset=data, batch_size=self.batch_size,
            sort=False, sort_within_batch=True,
            sort_key=lambda x: len(x.src),
            device=device, repeat=False)

        steps_per_epoch = len(batch_iterator)
        total_steps = steps_per_epoch * n_epochs

        step = start_step
        step_elapsed = 0
        for epoch in range(start_epoch, n_epochs + 1):
            log.debug("Epoch: %d, Step: %d" % (epoch, step))

            batch_generator = batch_iterator.__iter__()
            # consuming seen batches from previous training
            for _ in range((epoch - 1) * steps_per_epoch, step):
                next(batch_generator)

            model.train(True)
            for batch in batch_generator:
                step += 1
                step_elapsed += 1

                input_variables, input_lengths = getattr(batch, seq2seq.src_field_name)
                target_variables = getattr(batch, seq2seq.tgt_field_name)
                copy_mask = getattr(batch, 'extra', None)
                
                if copy_mask is not None:
                    model.copy_predictor.anneal()
                    
                    if input_variables.shape != copy_mask.shape:
                        v = data.fields[seq2seq.src_field_name].vocab
                        print(input_variables.shape, copy_mask.shape, input_variables[:,-1], copy_mask[:,-1])
                        for idx,d in enumerate(input_variables.data.cpu().numpy()):
                            print(copy_mask[idx])
                            print([v.itos[i] for i in d])
                
                loss = self._train_batch(input_variables, input_lengths.tolist(), target_variables, model, teacher_forcing_ratio, copy_mask)
                    
                # Record average loss
                print_loss_total += loss
                epoch_loss_total += loss

                if step % self.print_every == 0 and step_elapsed > self.print_every:
                    print_loss_avg = print_loss_total / self.print_every
                    print_loss_total = 0
                    if np.array(loss).shape != np.array(print_loss_total).shape:
                        print_loss_total = np.zeros_like(loss)
                    
                    log_msg = 'Progress: %d%%, Train %s: %s' % ( #%.4f
                        1. * step / total_steps * 100,
                        self.loss.name,
                        print_loss_avg)
                    log.info(log_msg)

                # Checkpoint
                if step % self.checkpoint_every == 0 or step == total_steps:
                    Checkpoint(model=model,
                               optimizer=self.optimizer,
                               epoch=epoch, step=step,
                               input_vocab=data.fields[seq2seq.src_field_name].vocab,
                               output_vocab=data.fields[seq2seq.tgt_field_name].vocab).save(self.expt_dir)

            if step_elapsed == 0: continue

            epoch_loss_avg = epoch_loss_total / min(steps_per_epoch, step - start_step)
            epoch_loss_total = 0
            if np.array(loss).shape != np.array(epoch_loss_total).shape:
                epoch_loss_total = np.zeros_like(loss)
            log_msg = "Finished epoch %d: Train %s: %s" % (epoch, self.loss.name, epoch_loss_avg) #%.4f
            if dev_data is not None:
                dev_loss, accuracy = self.evaluator.evaluate(model, dev_data)
                self.optimizer.update(dev_loss, epoch)
                log_msg += ", Dev %s: %s, Accuracy: %.4f" % (self.loss.name, dev_loss, accuracy) #%.4f
                model.train(mode=True)
            else:
                self.optimizer.update(epoch_loss_avg, epoch)

            log.info(log_msg)

