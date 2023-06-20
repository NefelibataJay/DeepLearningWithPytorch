import torch

from tool.common import remove_duplicates_and_blank
from tool.mask import *


def ctc_greedy_search(
        log_probs: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        eos: int = 2,
):
    beam_size = 1
    batch_size = log_probs.shape[0]

    maxlen = log_probs.size(1)

    # topk_index = log_probs.argmax(-1)
    topk_prob, topk_index = log_probs.topk(beam_size, dim=2)

    topk_index = topk_index.view(batch_size, maxlen)  # (B, maxlen)

    mask = make_pad_mask(encoder_out_lens, maxlen)  # (B, maxlen)

    topk_index = topk_index.masked_fill_(mask, eos)  # (B, maxlen)

    hyps = [hyp.tolist() for hyp in topk_index]

    scores = topk_prob.max(1)

    hyps = [remove_duplicates_and_blank(hyp) for hyp in hyps]

    return hyps, scores


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
