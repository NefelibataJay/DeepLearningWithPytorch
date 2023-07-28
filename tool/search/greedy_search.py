import torch

from tool.common import remove_duplicates_and_blank
from models.modules.mask import *


class GreedySearch:
    def __init__(self, max_length: int = 128, sos_id: int = 1, eos_id: int = 2, blank_id: int = 3, pad_id: int = 0):
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id

    def __call__(self, log_probs, output_lens, _type="ctc"):
        assert _type in ["ctc", "attention", "transducer"], "Decode_type Not Support!"
        if _type == "ctc":
            hyps, scores = self.ctc_greedy_search(log_probs, output_lens)
        elif _type == "attention":
            # TODO : attention greedy search
            pass
        elif _type == "transducer":
            # TODO : transducer greedy search
            pass

        return hyps, scores

    def _greedy_search(self, log_probs):
        """
        TODO add Batch
        Given a sequence, get the best path
            Args:
                log_probs (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
            Returns:
                List[int]: label Token
        """
        indices = torch.argmax(log_probs, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_id]
        return indices

    def ctc_greedy_search(self, log_probs: torch.Tensor, encoder_out_lens: torch.Tensor):
        batch_size = log_probs.shape[0]
        maxlen = log_probs.shape[1]

        # topk_index = log_probs.argmax(-1)
        topk_prob, topk_index = log_probs.topk(1, dim=2)

        topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

        mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)

        topk_index = topk_index.masked_fill_(mask, self.eos_id)  # (B, maxlen)

        hyps = [hyp.tolist() for hyp in topk_index]

        scores = topk_prob.max(1)

        hyps = [i for i in hyps if i != self.blank_id]  # remove blank
        # hyps = [remove_duplicates_and_blank(hyp, blank_id) for hyp in hyps]

        return hyps, scores

    def transducer_greedy_search(self, transducer, encoder_outputs, output_lengths):
        outputs = list()
        batch_size = encoder_outputs.size(0)

        decoder_inputs = encoder_outputs.new_zeros(batch_size, 1).fill_(self.sos_id).long()
        # first decode input is sos
        for i in range(self.max_length):
            decoder_outputs, hidden_states = transducer.predictor(decoder_inputs)
            # decoder_outputs (batch_size, i, decoder_output_size)



        return outputs

    def attention_greedy_search(self, encoder_outputs, encoder_output_lengths, decoder):
        pass


def transformer_greedy_search(decoder, encoder_outputs, encoder_output_lengths):
    beam_size = 1
    sos = 1
    eos = 2
    pad = 0
    batch_size = encoder_outputs.size(0)
    hidden_size = encoder_outputs.size(2)
    vocab_size = decoder.num_classes
    max_length = 128

    input_var = encoder_outputs.new_zeros(batch_size, 1).long()
    input_var = input_var.fill_(sos)  # add sos

    for di in range(1, max_length):
        input_lengths = torch.IntTensor(batch_size).fill_(di)
        dec_outputs, _, _ = decoder.forward_step(
            decoder_inputs=input_var,
            decoder_input_lengths=input_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=di, )
        # dec_outputs (batch_size, max_length, vocab_size)
        topk_prob, topk_index = decoder.fc(dec_outputs).log_softmax(dim=-1).topk(beam_size, dim=-1)
        # topk_index = decoder.fc(dec_outputs).log_softmax(dim=-1).argmax(dim=-1)
        # topk_index is token_id
        new_token_id = topk_index[:, -1, :]
        input_var = torch.cat([input_var, new_token_id], dim=1)

        if torch.all(new_token_id < eos + 1):
            break

    return input_var


def rnnt_greedy_search():
    pass
