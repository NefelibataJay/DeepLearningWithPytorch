class Search:
    def __init__(self, beam_size: int = 1, max_length: int = 100, sos_id: int = 1,
                 eos_id: int = 2, blank_id: int = 3, pad_id: int = 0):
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id
