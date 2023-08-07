import torch

from models.decoder.transformer_decoder import TransformerDecoder
from models.encoder.conformer_encoder import ConformerEncoder
from tool.loss import LabelSmoothingLoss, CTC


def test_conformer_ctc_attention():
    num_classes = 21

    input_dim = 80
    encoder_dim = 256
    num_encoder_layers = 12
    num_attention_heads = 4
    feed_forward_expansion_factor = 8
    conv_kernel_size = 15

    encoder = ConformerEncoder(input_dim=input_dim,
                               encoder_dim=encoder_dim,
                               num_layers=num_encoder_layers,
                               num_attention_heads=num_attention_heads,
                               feed_forward_expansion_factor=feed_forward_expansion_factor,
                               conv_kernel_size=conv_kernel_size, )

    attention_dim = 256
    attention_heads = 4
    linear_units = 2048
    num_layers = 6

    decoder = TransformerDecoder(vocab_size=num_classes,
                                 attention_dim=attention_dim,
                                 attention_heads=attention_heads,
                                 linear_units=linear_units,
                                 num_layers=num_layers, )

    fc = torch.nn.Linear(encoder_dim, num_classes, bias=False)

    criterion_att = LabelSmoothingLoss(
        size=num_classes,
        padding_idx=0,
        smoothing=0.1,
    )
    criterion_ctc = CTC(blank_id=11, reduction="mean")

    input_tensor = torch.rand(2, 80, 100)
    input_lengths = torch.LongTensor([100, 80])
    target_tensor = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [1, 2, 3, 4, 5, 10, 0, 0, 0, 0]])
    target_lengths = torch.LongTensor([10, 6])

    encoder_output, encoder_output_lengths = encoder(input_tensor.transpose(1,2), input_lengths)
    logits = fc(encoder_output)
    ctc_loss = criterion_ctc(logits,encoder_output_lengths, target_tensor, target_lengths)

    decoder_output, _ = decoder(encoder_output, encoder_output_lengths, target_tensor, target_lengths)

    att_loss = criterion_att(decoder_output, target_tensor)

    print(ctc_loss, att_loss)
    loss = 0.3 * ctc_loss + 0.7 * att_loss
    print(loss)


if __name__ == "__main__":
    test_conformer_ctc_attention()