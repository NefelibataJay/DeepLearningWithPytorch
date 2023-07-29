import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from models.encoder.conformer_encoder import ConformerEncoder
from tool.loss import CTC


class ConformerCTC(torch.nn.Module):
    def __init__(self, configs: DictConfig) -> None:
        super(ConformerCTC, self).__init__()
        self.configs = configs

        self.encoder_configs = self.configs.model.encoder
        self.num_classes = self.configs.model.num_classes
        self.sos_id = self.configs.tokenizer.sos_id
        self.eos_id = self.configs.tokenizer.eos_id
        self.pad_id = self.configs.tokenizer.pad_id
        self.blank_id = self.configs.tokenizer.blank_id

        self.encoder = ConformerEncoder(
            input_dim=self.encoder_configs.input_dim,
            encoder_dim=self.encoder_configs.encoder_dim,
            num_layers=self.encoder_configs.num_encoder_layers,
            num_attention_heads=self.encoder_configs.num_attention_heads,
            feed_forward_expansion_factor=self.encoder_configs.feed_forward_expansion_factor,
            input_dropout_p=self.encoder_configs.input_dropout_p,
            feed_forward_dropout_p=self.encoder_configs.feed_forward_dropout_p,
            attention_dropout_p=self.encoder_configs.attention_dropout_p,
            dropout_p=self.encoder_configs.dropout_p,
            conv_kernel_size=self.encoder_configs.conv_kernel_size,
            half_step_residual=self.encoder_configs.half_step_residual,
        )

        self.fc = Linear(self.encoder_configs.encoder_dim, self.num_classes, bias=False)

        self.ctc_criterion = CTC(blank_id=self.blank_id, reduction="mean")

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Tensor, target_lengths: Tensor):
        result = dict()
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs)

        loss = self.ctc_criterion(logits, output_lengths, targets, target_lengths)
        result["loss"] = loss
        result["logits"] = logits
        result["output_lengths"] = output_lengths
        result["encoder_outputs"] = encoder_outputs

        return result
