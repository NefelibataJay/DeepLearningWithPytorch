import torch
from omegaconf import DictConfig
from torch import Tensor
from torch.nn import Linear

from models.decoder.transformer_decoder import TransformerDecoder
from models.encoder.conformer_encoder import ConformerEncoder
from tool.common import add_sos, add_eos
from tool.loss import CTC, LabelSmoothingLoss


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

        self.ctc_weight = self.configs.weight_conf.ctc_weight
        self.lsm_weight = self.configs.weight_conf.lsm_weight

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

        self.criterion_att = LabelSmoothingLoss(
            size=self.num_classes,
            padding_idx=0,
            smoothing=self.lsm_weight,
        )

    def forward(self, inputs: Tensor, input_lengths: Tensor, targets: Tensor, target_lengths: Tensor):
        """
        Args:
            inputs: (batch, max_seq_len, feat_dim)
            input_lengths: (batch)
            targets: (batch, max_seq_len)  # padded , Not SOS and EOS
            target_lengths: (batch)
        Returns:
            result: dict
        """
        result = dict()
        encoder_outputs, output_lengths = self.encoder(inputs, input_lengths)

        # calculate ctc loss
        logits = self.fc(encoder_outputs)
        result["logits"] = logits

        ctc_loss = self.ctc_criterion(logits, output_lengths, targets, target_lengths)
        result["ctc_loss"] = ctc_loss

        ys_in_pad = add_sos(targets, self.sos_id, self.pad_id)  # (batch, max_seq_len + 1)
        ys_in_lens = target_lengths + 1

        decoder_outputs, _ = self.decoder(encoder_outputs, output_lengths, ys_in_pad, ys_in_lens)

        # calculate attention loss
        ys_out_pad = add_eos(targets, self.eos_id, self.pad_id)  # (batch, max_seq_len + 1)

        att_loss = self.criterion_att(decoder_outputs, ys_out_pad)
        result["att_loss"] = att_loss

        result['loss'] = self.ctc_weight * ctc_loss + (1 - self.ctc_weight) * att_loss

        return result
