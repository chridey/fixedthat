import torch.nn.functional as F

from seq2seq.models import Seq2seq

class CustomSeq2seq(Seq2seq):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax, predict_size=False, copy_predictor=None):
        super(CustomSeq2seq, self).__init__(encoder, decoder, decode_function=F.log_softmax)
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.predict_size = predict_size
        self.copy_predictor = copy_predictor
        
    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        if getattr(self.decoder, 'rnn', None) is not None:
            self.decoder.rnn.flatten_parameters()        
        
    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, counts=None, mask=None, filter_illegal=True, use_prefix=False):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        if self.predict_size:
            result = self.decoder(encoder_outputs)
        else:
            if self.copy_predictor is not None:
                copy_mask, _, _ = self.copy_predictor(encoder_outputs, mask)
            else:
                copy_mask = mask
            result = self.decoder(input_variable,
                                decoder_inputs=target_variable,
                                encoder_hidden=encoder_hidden,
                                encoder_outputs=encoder_outputs,
                                function=self.decode_function,
                                teacher_forcing_ratio=teacher_forcing_ratio,
                                counts=counts,
                                mask=mask,
                                copy_mask=copy_mask,
                                filter_illegal=filter_illegal,
                                use_prefix=use_prefix)
            if self.copy_predictor is not None:
                result[-1]['copy_mask'] = copy_mask
        return result
                            
