from typing import Optional, List, Tuple

import torch
from torch import nn, Tensor

RNN_TYPE = {
    'lstm': nn.LSTM,
    'gru': nn.GRU,
    'rnn': nn.RNN,
}


class TransducerJoint(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 enc_output_size: int = 256,
                 pred_output_size: int = 256, ):
        super().__init__()
        # TODO optional for pre_fc

        self.joint_dim = enc_output_size + pred_output_size

        self.f1 = nn.Linear(self.joint_dim, self.joint_dim, bias=True)
        self.tanh = nn.Tanh()
        self.f2 = nn.Linear(self.joint_dim, vocab_size, bias=True)

    def forward(self, encoder_outputs: Tensor, decoder_outputs: Tensor):
        """
        Args:
            encoder_outputs (torch.Tensor): [B, input_length, encoder_output_size]
            decoder_outputs (torch.Tensor): [B, target_length, pred_output_size]
        Return:
            [B,input_length,target_length,joint_dim]
        """
        input_length = encoder_outputs.size(1)
        target_length = decoder_outputs.size(1)

        encoder_outputs = encoder_outputs.unsqueeze(2)
        decoder_outputs = decoder_outputs.unsqueeze(1)

        encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
        decoder_outputs = decoder_outputs.repeat([1, input_length, 1, 1])

        outputs = torch.cat((encoder_outputs, decoder_outputs), dim=-1)
        logits = self.fc(outputs).log_softmax(dim=-1)

        return logits


class RnnPredictor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 embed_size: int,
                 hidden_size: int = 1048,
                 output_size: int = 256,
                 num_layers: int = 1,
                 embed_dropout: float = 0.1,
                 rnn_dropout: float = 0.1,
                 rnn_type: str = "lstm",
                 bias: bool = True,
                 pad: int = 0,
                 ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size, padding_idx=pad)
        self.dropout = nn.Dropout(embed_dropout)

        self.n_layers = num_layers
        self.hidden_size = hidden_size

        self.rnn = RNN_TYPE[rnn_type](
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=True,
            dropout=rnn_dropout,
            bidirectional=False,
        )

        self.final_projection = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: Tensor, input_lengths: Optional[List[torch.Tensor]] = None,
                hidden_states: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward  `inputs` (targets) for training.

        Args:
            inputs : [batch, target_length).
            input_lengths : [batch]
            hidden_states : Previous hidden states.
        """
        embed = self.dropout(self.embed(inputs))

        if hidden_states is not None:
            outputs, hidden_states = self.rnn(embed, hidden_states)
        else:
            outputs, hidden_states = self.rnn(embed)

        outputs = self.final_projection(outputs)
        return outputs, hidden_states
