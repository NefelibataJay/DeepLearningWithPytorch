import torch

from tool.common import remove_duplicates_and_blank
from models.modules.mask import *
from tool.search.base_search import Search


class GreedySearch(Search):
    def __init__(self, max_length, sos_id: int = 1, eos_id: int = 2, blank_id: int = 3, pad_id: int = 0):
        super(GreedySearch, self).__init__(max_length=max_length, sos_id=sos_id, eos_id=eos_id, blank_id=blank_id,
                                           pad_id=pad_id)

    def __call__(self, log_probs,output_lens, decode_type="ctc"):
        assert decode_type in ["ctc", "attention", "transducer"], "Decode_type Not Support!"
        if decode_type == "ctc":
            hyps, scores = self.ctc_greedy_search(log_probs, output_lens)
        elif decode_type == "attention":
            # TODO : attention greedy search
            pass
        elif decode_type == "transducer":
            # TODO : transducer greedy search
            pass

        return hyps, scores

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

        hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]

        return hyps, scores


def transducer_greedy_search(transducer_model, encoder_inputs, max_length: int = 128):
    """
    transducer_model : Transducer
    encoder_inputs : (batch_size, max_length, dim)
    """
    outputs = list()
    encoder_outputs, output_lengths = transducer_model.encoder(encoder_inputs)
    for encoder_output in encoder_outputs:
        pred_tokens = list()
        decoder_input = encoder_output.new_zeros(1, 1).fill_(transducer_model.decoder.sos_id).long()
        decoder_output, hidden_state = transducer_model.decoder(decoder_input)

        for t in range(max_length):
            step_output = transducer_model.joint(encoder_output[t].view(-1), decoder_output.view(-1))

            pred_token = step_output.argmax(dim=0)
            pred_token = int(pred_token.item())
            pred_tokens.append(pred_token)

            decoder_input = torch.LongTensor([[pred_token]])
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()

            decoder_output, hidden_state = transducer_model.decoder(decoder_input, hidden_states=hidden_state)

        outputs.append(torch.LongTensor(pred_tokens))
    return torch.stack(outputs, dim=0)


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
