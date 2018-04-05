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

from fixedthat.modeling.fields import CustomTargetField as TargetField
from fixedthat.modeling.customDecoderRNN import CustomDecoderRNN
from fixedthat.modeling.customSeq2seq import CustomSeq2seq
from fixedthat.modeling.customPredictor import CustomPredictor
from fixedthat.modeling.loss import PoissonLoss, MSELoss
from fixedthat.modeling.sizePredictor import SizePredictor
from fixedthat.modeling.beamSearchDecoder import BeamSearchDecoder

from fixedthat.modeling.customEvaluator import Evaluator
from fixedthat.modeling.customTrainer import CustomTrainer as SupervisedTrainer

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

opt = parser.parse_args()

LOG_FORMAT = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'
logging.basicConfig(format=LOG_FORMAT, level=getattr(logging, opt.log_level.upper()))
logging.info(opt)

if opt.load_checkpoint is not None:
    logging.info("loading checkpoint from {}".format(os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)))
    checkpoint_path = os.path.join(opt.expt_dir, Checkpoint.CHECKPOINT_DIR_NAME, opt.load_checkpoint)
    checkpoint = Checkpoint.load(checkpoint_path)
    seq2seq = checkpoint.model
    input_vocab = checkpoint.input_vocab
    output_vocab = checkpoint.output_vocab
else:
    # Prepare dataset
    src = SourceField()
    if opt.size:
        tgt = TargetField(get_size=True)
    else:
        tgt = TargetField() #counter=True) #add_start=False)
    max_len = 50
    def len_filter(example):
        try:
            return len(example.src) <= max_len and len(example.tgt) <= max_len
        except AttributeError:
            print(example.__dict__)
            return True
        
    train = torchtext.data.TabularDataset(
        path=opt.train_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    dev = torchtext.data.TabularDataset(
        path=opt.dev_path, format='tsv',
        fields=[('src', src), ('tgt', tgt)],
        filter_pred=len_filter
    )
    src.build_vocab(train, max_size=50000)
    #TODO if args.counter
    #tgt.build_vocab([zip(*x)[0] for x in train.tgt], max_size=50000)
    #else
    tgt.build_vocab(train, max_size=50000)
    input_vocab = src.vocab
    output_vocab = tgt.vocab

    # NOTE: If the source field name and the target field name
    # are different from 'src' and 'tgt' respectively, they have
    # to be set explicitly before any training or inference
    # seq2seq.src_field_name = 'src'
    # seq2seq.tgt_field_name = 'tgt'

    # Prepare loss
    weight = torch.ones(len(tgt.vocab))
    pad = tgt.vocab.stoi[tgt.pad_token]
    #TODO: if args.counter
    #loss = CounterLoss(Perplexity(weight, pad))

    loss = Perplexity(weight, pad)
    if opt.size:
        #loss = PoissonLoss()#log_input=True)
        loss = MSELoss()
    
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

        if opt.counter:
            decoder_type = CustomDecoderRNN
            seq2seq_type = CustomSeq2seq
            decoder = decoder_type(src.vocab, tgt.vocab, max_len, hidden_size * 2 if bidirectional else hidden_size,
                                   dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                   eos_id=tgt.eos_id, sos_id=tgt.sos_id)
        elif opt.size:
            decoder = SizePredictor(hidden_size * 2 if bidirectional else hidden_size, hidden_size,
                                      bidirectional=bidirectional, use_attention=True)
            seq2seq_type = CustomSeq2seq            
        else:
            decoder = decoder_type(len(tgt.vocab), max_len, hidden_size * 2 if bidirectional else hidden_size,
                                   dropout_p=0.2, use_attention=True, bidirectional=bidirectional,
                                   eos_id=tgt.eos_id, sos_id=tgt.sos_id)

        seq2seq = seq2seq_type(encoder, decoder, predict_size=opt.size)
        if torch.cuda.is_available():
            seq2seq.cuda()

        for param in seq2seq.parameters():
            param.data.uniform_(-0.08, 0.08)

        # Optimizer and learning rate scheduler can be customized by
        # explicitly constructing the objects and pass to the trainer.
        #
        # optimizer = Optimizer(torch.optim.Adam(seq2seq.parameters()), max_grad_norm=5)
        # scheduler = StepLR(optimizer.optimizer, 1)
        # optimizer.set_scheduler(scheduler)

    # train
    evaluator = Evaluator(loss=loss, batch_size=100, prediction_dir=opt.expt_dir)
    t = SupervisedTrainer(loss=loss, batch_size=100, #32,
                          checkpoint_every=1000,
                          print_every=10, expt_dir=opt.expt_dir,
                          evaluator=evaluator)

    seq2seq = t.train(seq2seq, train,
                      num_epochs=opt.num_epochs, dev_data=dev,
                      optimizer=optimizer,
                      teacher_forcing_ratio=0.5,
                      resume=opt.resume)

if opt.counter:
    seq2seq.decoder = BeamSearchDecoder(seq2seq.decoder, 10)
    predictor = CustomPredictor(seq2seq, input_vocab, output_vocab)
else:
    predictor = Predictor(seq2seq, input_vocab, output_vocab)
    
while True:
    seq_str = raw_input("Type in a source sequence:")
    seq = seq_str.strip().split()
    if opt.counter:
        counter = int(raw_input("Enter the number of new words to generate (must be an integer):"))
        print(predictor.predict(seq, counter))
    else:
        print(predictor.predict(seq))
