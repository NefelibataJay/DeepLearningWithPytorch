import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from models.decoder.transformer_decoder import TransformerDecoder
from models.encoder.conformer_encoder import ConformerEncoder
from tool.loss import CTC


class ConformerCTCAttention(torch.nn.Module):
    def __init__(self, configs: DictConfig) -> None:
        super(ConformerCTCAttention, self).__init__()
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
            conv_expansion_factor=self.encoder_configs.conv_expansion_factor,
            input_dropout_p=self.encoder_configs.input_dropout_p,
            feed_forward_dropout_p=self.encoder_configs.feed_forward_dropout_p,
            attention_dropout_p=self.encoder_configs.attention_dropout_p,
            conv_dropout_p=self.encoder_configs.conv_dropout_p,
            conv_kernel_size=self.encoder_configs.conv_kernel_size,
            half_step_residual=self.encoder_configs.half_step_residual,
        )

        self.fc = Linear(self.encoder_configs.encoder_dim, self.num_classes, bias=False)

        self.ctc_criterion = CTC(blank_id=self.blank_id, reduction="mean")

        self.decoder_configs = self.configs.model.decoder
        self.decoder = TransformerDecoder(
            vocab_size=self.num_classes,
            attention_dim=self.decoder_configs.attention_dim,
            attention_heads=self.decoder_configs.attention_heads,
            linear_units=self.decoder_configs.linear_units,
            num_layers=self.decoder_configs.num_layers,
            dropout_rate=self.decoder_configs.dropout_rate,
            positional_dropout_rate=self.decoder_configs.positional_dropout_rate,
            self_attention_dropout_rate=self.decoder_configs.self_attention_dropout_rate,
            src_attention_dropout_rate=self.decoder_configs.src_attention_dropout_rate,

        )


    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Tensor, target_lengths: Tensor):
        result = dict()
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)
        logits = self.fc(encoder_outputs)

        ctc_loss = self.ctc_criterion(logits, targets, output_lengths, target_lengths)
        result["ctc_loss"] = ctc_loss

        return result
