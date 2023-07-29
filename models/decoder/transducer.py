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
                 pred_output_size: int = 256,
                 joint_dim: int = 512,
                 joint_type: str = "add", ):
        super().__init__()
        assert joint_type in ["add", "concat"], "joint_type must be in ['add', 'concat']"
        self.joint_type = joint_type

        self.lin_enc = torch.nn.Linear(enc_output_size, joint_dim)
        self.lin_dec = torch.nn.Linear(pred_output_size, joint_dim)

        if joint_type == "concat":
            self.f1 = nn.Linear(enc_output_size + pred_output_size, joint_dim)

        self.activation = nn.Tanh()  # TODO : support other activation function
        self.out_pro = nn.Linear(joint_dim, vocab_size)

    def forward(self, encoder_outputs: Tensor, predictor_outputs: Tensor):
        """
        Args:
            encoder_outputs (torch.Tensor): [B, input_length, encoder_output_size]
            predictor_outputs (torch.Tensor): [B, target_length, pred_output_size]
        Return:
            [B,input_length,target_length,joint_dim]
        """
        outputs = None

        input_length = encoder_outputs.size(1)
        target_length = predictor_outputs.size(1)

        enc_out = self.lin_enc(encoder_outputs)
        pred_out = self.lin_dec(predictor_outputs)
        enc_out = enc_out.unsqueeze(2)  # [B, input_length, 1, joint_dim]
        pred_out = pred_out.unsqueeze(1)  # [B, 1, target_length, joint_dim]

        if self.joint_type == "add":
            outputs = enc_out + pred_out  # [B,input_length,target_length,joint_dim]

        elif self.joint_type == "concat":
            encoder_outputs = encoder_outputs.repeat([1, 1, target_length, 1])
            pred_out = pred_out.repeat([1, input_length, 1, 1])

            outputs = torch.cat((encoder_outputs, pred_out), dim=-1)
            outputs = self.f1(outputs)

        logits = self.out_pro(self.activation(outputs))

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
            batch_first=True,
            dropout=rnn_dropout,
        )

        self.final_projection = nn.Linear(hidden_size, output_size)

    def forward(self, inputs: Tensor, hidden_states=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward  `inputs` (targets) for training.
        Args:
            inputs : (batch, target_max_length). labels
            hidden_states : (batch, hidden_size). hidden states
        Returns:
            outputs : (batch, target_max_length, pred_output_size). Decoder outputs.
        """
        embed = self.dropout(self.embed(inputs))

        if hidden_states is not None:
            outputs, hidden_states = self.rnn(embed, hidden_states)
        else:
            outputs, hidden_states = self.rnn(embed)

        outputs = self.final_projection(outputs)
        return outputs, hidden_states
