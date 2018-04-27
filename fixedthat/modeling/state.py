class HiddenState:
    '''
    class for storing any state variables that we need to pass to the next decode timestep
    '''
    def __init__(self, decoder_hidden, *state_variables):
        self._decoder_hidden = decoder_hidden
        self._fields = [self._decoder_hidden] + list(state_variables)

    @property
    def decoder_hidden(self):
        return self._decoder_hidden
        
    @property
    def fields(self):
        return self._fields
