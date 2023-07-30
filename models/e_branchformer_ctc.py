import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from models.encoder.e_branchformer_encoder import EBranchformerEncoder
from tool.loss import CTC


class BranchformerCTC(torch.nn.Module):
    def __init__(self, configs: DictConfig):
        super(BranchformerCTC, self).__init__()
        self.configs = configs

        self.encoder_configs = self.configs.model.encoder
        self.num_classes = self.configs.model.num_classes
        self.sos_id = self.configs.tokenizer.sos_id
        self.eos_id = self.configs.tokenizer.eos_id
        self.pad_id = self.configs.tokenizer.pad_id
        self.blank_id = self.configs.tokenizer.blank_id

        self.encoder = EBranchformerEncoder(
            input_dim=self.encoder_configs.input_dim,
            encoder_dim=self.encoder_configs.encoder_dim,
            num_layers=self.encoder_configs.num_layers,
            attention_heads=self.encoder_configs.attention_heads,
            feed_forward_expansion_factor=self.encoder_configs.feed_forward_expansion_factor,
            cgmlp_linear_expansion_factor=self.encoder_configs.cgmlp_linear_expansion_factor,
            cgmlp_conv_kernel_size=self.encoder_configs.cgmlp_conv_kernel_size,
            dropout_rate=self.encoder_configs.dropout_rate,
            positional_dropout_rate=self.encoder_configs.positional_dropout_rate,
            attention_dropout_rate=self.encoder_configs.attention_dropout_rate,
            feed_forward_dropout_rate=self.encoder_configs.feed_forward_dropout_rate,
            cgmlp_conv_dropout_rate=self.encoder_configs.cgmlp_conv_dropout_rate,
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
