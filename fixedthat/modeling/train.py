from __future__ import print_function

import sys
import os
import argparse
import logging

import torch
from torch.optim.lr_scheduler import StepLR
import torchtext

import seq2seq
from seq2seq.trainer import SupervisedTrainer
from seq2seq.models import EncoderRNN, DecoderRNN, Seq2seq
from seq2seq.loss import Perplexity
from seq2seq.optim import Optimizer
from seq2seq.dataset import SourceField, TargetField
from seq2seq.evaluator import Predictor
from seq2seq.util.checkpoint import Checkpoint

from fixedthat.modeling.fields import CustomTargetField
from fixedthat.modeling.customDecoderRNN import CustomDecoderRNN
from fixedthat.modeling.customSeq2seq import CustomSeq2seq
from fixedthat.modeling.customPredictor import CustomPredictor
from fixedthat.modeling.loss import PoissonLoss, MSELoss, BCELoss, MTLLoss
from fixedthat.modeling.sizePredictor import SizePredictor
from fixedthat.modeling.copyPredictor import CopyPredictor
from fixedthat.modeling.beamSearchDecoder import BeamSearchDecoder

from fixedthat.modeling.customEvaluator import Evaluator
from fixedthat.modeling.customTrainer import CustomTrainer as SupervisedTrainer

from fixedthat.modeling.transducer import Transducer

try:
    raw_input          # Python 2
except NameError:
    raw_input = input  # Python 3

# Sample usage:
#     # training
#     python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH
#     # resuming from the latest checkpoint of the experiment
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --resume
#      # resuming from a specific checkpoint
#      python examples/sample.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --expt_dir $EXPT_PATH --load_checkpoint $CHECKPOINT_DIR

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path',
                    help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path',
                    help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume',
                    default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level',
                    default='info',
                    help='Logging level.')
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--counter', action='store_true',
                    help='whether to add the counter for newly generated content')
parser.add_argument('--size', action='store_true',
                    help='whether to predict the size of the set of words in content that are not in input')
parser.add_argument('--transducer', action='store_true',
                    help='whether to transducer')
parser.add_argument('--translate', action='store_true',
                    help='whether to switch to translate mode after loading model, if test_path is also provided the model will write translated sentences to the file path stored in translate_path')
parser.add_argument('--test_path', action='store', dest='test_path',
                    help='Path to test data')
parser.add_argument('--translate_path', action='store', dest='translate_path',
                    help='Path to translate data')
parser.add_argument('--max_range', type=int, default=5,
                    help='for translation with counter, the range of values to test')
parser.add_argument('--count_hidden_size', type=int, default=15)
parser.add_argument('--verbose', action='store_true')
parser.add_argument('--copy_predict', action='store_true')
parser.add_argument('--no_text', action='store_true')
parser.add_argument('--filter_illegal', action='store_true')
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--beam_width', type=int, default=10)
parser.add_argument('--use_prefix', action='store_true')

opt = parser.parse_args()

multi_task = False
if (opt.copy_predict or opt.size) and not opt.no_text:
    multi_task = True
    
LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

# Prepare dataset
src = SourceField()

extra_tgt = None
if opt.copy_predict:
    extra_tgt = torchtext.data.Field(use_vocab=False, batch_first=True, preprocessing=lambda x:map(int, x),
                               pad_token=2) #UNDO , init_token=2)
elif opt.size:
    #tgt = torchtext.data.Field(sequential=False, use_vocab=False, batch_first=True)
    extra_tgt = CustomTargetField(get_size=True)

tgt = None
if not opt.no_text:
    tgt = TargetField() #counter=True) #add_start=False)

used_fields=[('src', src)]
if opt.no_text:
    used_fields.append(('tgt', extra_tgt))
else:
    used_fields.append(('tgt', tgt))
    if extra_tgt is not None:
        used_fields.append(('extra', extra_tgt))

        
max_len = 50 
def len_filter(example):
    try:
        return len(example.src) <= max_len and len(example.tgt) <= max_len
    except AttributeError:
        print('Problem in len_filter, fields are ', example.__dict__)
        return True

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:

    
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        #fields=[('src', src), ('tgt', tgt)],
        fields=used_fields,
        filter_pred=len_filter
    )

    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        #fields=[('src', src), ('tgt', tgt)],
        fields=used_fields,
        filter_pred=len_filter
    )
    src.build_vocab(train, min_freq=4) #max_size=50000)
    input_vocab = src.vocab
    #TODO if args.counter
    #tgt.build_vocab([zip(*x)[0] for x in train.tgt], max_size=50000)
    #else
    if not opt.no_text: #opt.copy_predict and not opt.size:
        tgt.build_vocab(train, min_freq=4) #max_size=50000)
        output_vocab = tgt.vocab
    else:
        tgt.vocab = None
        output_vocab = None

    print(len(input_vocab))
    print(len(output_vocab))
    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    if output_vocab is not None:
        weight = torch.ones(len(tgt.vocab))
        pad = tgt.vocab.stoi[tgt.pad_token]
    #TODO: if args.counter
    #loss = CounterLoss(Perplexity(weight, pad))

    #TODO: combined loss
    if multi_task:
        loss = MTLLoss([Perplexity(weight, pad), BCELoss(pad=2)], lambdas=[1,10]) #Perplexity(weight, pad) #
    elif opt.size:
        #loss = PoissonLoss()#log_input=True)
        loss = MSELoss()
    elif opt.copy_predict:
        loss = BCELoss()
    else:
        loss = Perplexity(weight, pad)
        
    if torch.cuda.is_available():
        loss.cuda()

    seq2seq = None
    optimizer = None
    if not opt.resume:
        # Initialize model
        hidden_size=300
        bidirectional = True

        encoder_type = EncoderRNN
        decoder_type = DecoderRNN
        seq2seq_type = Seq2seq
        
        encoder = encoder_type(len(src.vocab), max_len, hidden_size,
                               bidirectional=bidirectional, variable_lengths=True)

        copy_predictor=None
        if opt.copy_predict:
            copy_predictor = CopyPredictor(hidden_size * 2 if bidirectional else hidden_size, hidden_size,
                                            bidirectional=bidirectional, use_attention=True)
        if opt.counter:
            decoder_type = CustomDecoderRNN
            seq2seq_type = CustomSeq2seq
            decoder = decoder_type(src.vocab, tgt.vocab, max_len, hidden_size * 2 if bidirectional else hidden_size,
                                   dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                   eos_id=tgt.eos_id, sos_id=tgt.sos_id, count_hidden_size=opt.count_hidden_size,
                                   copy_predictor=opt.copy_predict)
        elif opt.size:
            decoder = SizePredictor(hidden_size * 2 if bidirectional else hidden_size, hidden_size,
                                      bidirectional=bidirectional, use_attention=True)
            seq2seq_type = CustomSeq2seq
        elif opt.copy_predict:
            decoder = copy_predictor
            seq2seq_type = CustomSeq2seq            
        elif opt.transducer:
            seq2seq_type = CustomSeq2seq            
            decoder = Transducer(src.vocab, tgt.vocab, max_len, hidden_size * 2 if bidirectional else hidden_size,
                                   dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                   eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        else:
            decoder = decoder_type(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                                   dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                   eos_id=tgt.eos_id, sos_id=tgt.sos_id)

        if opt.size or opt.copy_predict:
            seq2seq = seq2seq_type(encoder, decoder, predict_size=(not multi_task),
                                   copy_predictor=(copy_predictor if multi_task else None))
        else:
            seq2seq = seq2seq_type(encoder, decoder)
            
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        scheduler = StepLR(optimizer.optimizer, 1)
        optimizer.set_scheduler(scheduler)

    # train
    #TODO: fix pad for multi task case
    evaluator = Evaluator(loss=loss, batch_size=opt.batch_size,
                          pad=2 if opt.copy_predict and opt.no_text else None,
                          filter_illegal=opt.filter_illegal,
                          use_prefix=opt.use_prefix) #, prediction_dir=opt.expt_dir)
    t = SupervisedTrainer(loss=loss, batch_size=opt.batch_size, #32,
                          checkpoint_every=1000,
                          print_every=10, expt_dir=opt.expt_dir,
                          evaluator=evaluator,
                          filter_illegal=opt.filter_illegal,
                          use_prefix=opt.use_prefix)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=opt.num_epochs, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

if opt.translate:    
    #if opt.counter or opt.transducer:
    if multi_task or (not opt.size and not opt.copy_predict):
        seq2seq.decoder = BeamSearchDecoder(seq2seq.decoder, opt.beam_width)
        predictor = CustomPredictor(seq2seq, input_vocab, output_vocab)
    else:
        predictor = CustomPredictor(seq2seq)
    #else:
    #predictor = Predictor(seq2seq, input_vocab, output_vocab)

    if opt.translate_path:
        f = open(opt.translate_path, 'w')
    else:
        f = sys.stdout

    if opt.test_path:        
        test = torchtext.data.TabularDataset(
            path=opt.test_path, format='tsv',
            fields=[('idx', torchtext.data.RawField())] + used_fields,
            #filter_pred=len_filter
        )
        src.vocab = input_vocab
        tgt.vocab = output_vocab
        
        if opt.counter:
            output = predictor.predict_batch(test, opt.batch_size/opt.beam_width,
                                             file_handle=f, use_counter=opt.counter) #start=1, end=opt.max_range)
        else:
            output = predictor.predict_batch(test, opt.batch_size/opt.beam_width,
                                             file_handle=f, use_counter=opt.counter)

        #for seq in output:
        #    print(' '.join(seq), file=f)
    else:
        while True:
            seq_str = raw_input("Type in a source sequence:")
            seq = seq_str.strip().split()
            if opt.counter:
                counter = int(raw_input("Enter the number of new words to generate (must be an integer):"))
                print(predictor.predict(seq, counter, verbose=opt.verbose), file=f)
            else:
                print(predictor.predict(seq, verbose=opt.verbose), file=f)
