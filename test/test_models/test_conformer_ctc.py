import torch

from models.encoder.conformer_encoder import ConformerEncoder
from tool.loss import CTC


def test_conformer_ctc():
    input_dim = 80
    encoder_dim = 16
    num_encoder_layers = 2
    num_attention_heads = 4
    feed_forward_expansion_factor = 4
    num_classes = 15
    blank_id = 3
    batch_size = 2
    encoder = ConformerEncoder(
        input_dim=input_dim,
        encoder_dim=encoder_dim,
        num_layers=num_encoder_layers,
        num_attention_heads=num_attention_heads,
        feed_forward_expansion_factor=feed_forward_expansion_factor,
    )
    fc = torch.nn.Linear(encoder_dim, num_classes, bias=False)
    ctc_criterion = CTC(blank_id=blank_id, reduction="mean")
    print(encoder)

    inputs = torch.randn(batch_size, 100, input_dim)
    input_lengths = torch.LongTensor([100, 80])
    outputs, output_lengths = encoder(inputs, input_lengths)
    logits = fc(outputs)

    targets = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                                [1, 2, 3, 4, 5, 6, 0, 0, 0, 0]])
    target_lengths = torch.LongTensor([10, 6])
    loss = ctc_criterion(logits,output_lengths, targets, target_lengths)

    print(loss)


if __name__ == '__main__':
    test_conformer_ctc()