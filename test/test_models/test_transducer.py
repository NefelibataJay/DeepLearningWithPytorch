from models.decoder.transducer import RnnPredictor
import torch


def test_transducer():
    rnn = RnnPredictor(vocab_size=10, embed_size=10, hidden_size=20, output_size=16, num_layers=2)
    print(rnn)
    x = torch.rand(2, 10).to(torch.int)  # [B, seq_len]
    y, hidden_states = rnn(x)
    print(y.shape)


def init_state(batch_size: int):
    """Initialize decoder states.

    Args:
        batch_size: Batch size.

    Returns:
        : Initial decoder hidden states. ((N, B, D_dec), (N, B, D_dec))

    """
    hidden_size = 10
    num_layers = 2
    h_n = torch.zeros(
        num_layers,
        batch_size,
        hidden_size,
    )

    c_n = torch.zeros(
        num_layers,
        batch_size,
        hidden_size,
    )

    return (h_n, c_n)


if __name__ == "__main__":
    test_transducer()
