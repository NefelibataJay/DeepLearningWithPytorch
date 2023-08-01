import torch

from models.decoder.transformer_decoder import TransformerDecoder
from models.encoder.conformer_encoder import ConformerEncoder
from tool.common import add_sos, add_eos
from tool.loss import CTC, LabelSmoothingLoss
from tool.search.beam_search import BeamSearch


class B:
    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder


def test_transformer_decoder():
    input_dim = 80
    encoder_dim = 16
    attention_dim = 16
    num_encoder_layers = 2
    num_attention_heads = 4
    feed_forward_expansion_factor = 4
    num_classes = 15
    sos_id = 1
    eos_id = 2
    blank_id = 3
    batch_size = 2
    attention_heads = 4
    encoder = ConformerEncoder(
        input_dim=input_dim,
        encoder_dim=encoder_dim,
        num_layers=num_encoder_layers,
        num_attention_heads=num_attention_heads,
        feed_forward_expansion_factor=feed_forward_expansion_factor,
    )
    decoder = TransformerDecoder(
        vocab_size=num_classes,
        attention_dim=attention_dim,
        attention_heads=attention_heads,
        linear_units=4 * attention_dim,
        num_layers=num_encoder_layers,
    )

    # inputs = torch.randn(batch_size, 100, input_dim)
    # input_lengths = torch.LongTensor([100, 80])
    # encoder_outputs, output_lengths = encoder(inputs, input_lengths)
    #
    # targets = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    #                             [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]])
    # target_lengths = torch.LongTensor([10, 6])
    #
    # ys_in_pad = add_sos(targets, sos_id, 0)  # (batch, max_seq_len + 1)
    # ys_in_lens = target_lengths + 1
    #
    # decoder_outputs, _ = decoder(encoder_outputs, output_lengths, ys_in_pad, ys_in_lens)
    #
    # criterion_att = LabelSmoothingLoss(
    #     size=num_classes,
    #     padding_idx=0,
    # )
    #
    # ys_out_pad = add_eos(targets, eos_id, 0)  # (batch, max_seq_len + 1)
    #
    # loss = criterion_att(decoder_outputs, ys_out_pad)
    #
    # print(loss)

    # decoding
    beam_search = BeamSearch()
    batch_size = 1
    speech = torch.randn(batch_size, 20, input_dim)
    speech_lengths = torch.LongTensor([20])
    model = B(encoder,decoder)
    hyps, _ = beam_search.attention_beam_search(speech, speech_lengths, model)

    print(hyps)


if __name__ == '__main__':
    test_transformer_decoder()
