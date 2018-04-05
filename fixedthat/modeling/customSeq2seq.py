import torch.nn.functional as F

from seq2seq.models import Seq2seq

class CustomSeq2seq(Seq2seq):
    def __init__(self, encoder, decoder, decode_function=F.log_softmax, predict_size=False):
        super(CustomSeq2seq, self).__init__(encoder, decoder, decode_function=F.log_softmax)
        self.encoder = encoder
        self.decoder = decoder
        self.decode_function = decode_function
        self.predict_size = predict_size
        
    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0, counts=None):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)

        if self.predict_size:
            result = self.decoder(encoder_outputs)
        else:
            result = self.decoder(input_variable,
                                decoder_inputs=target_variable,
                                encoder_hidden=encoder_hidden,
                                encoder_outputs=encoder_outputs,
                                function=self.decode_function,
                                teacher_forcing_ratio=teacher_forcing_ratio,
                                counts=counts)
        return result
                            
