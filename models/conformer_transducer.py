import torch
import torchaudio
from omegaconf import DictConfig
from torchaudio.transforms import RNNTLoss

from models.decoder.transducer import TransducerJoint, RnnPredictor
from models.encoder.conformer_encoder import ConformerEncoder
from tool.common import add_blank


class ConformerTransducer(torch.nn.Module):
    def __init__(self, configs: DictConfig) -> None:
        super(ConformerTransducer, self).__init__()
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

        self.joint_configs = self.configs.model.joint

        self.joint = TransducerJoint(
            vocab_size=self.num_classes,
            enc_output_size=self.joint_configs.encoder_dim,
            pred_output_size=self.joint_configs.predictor_dim,
        )

        self.predictor_config = self.configs.model.predictor

        self.predictor = RnnPredictor(
            vocab_size=self.num_classes,
            embed_size=self.predictor_config.embed_size,
            hidden_size=self.predictor_config.hidden_size,
            output_size=self.predictor_config.predictor_dim,
            num_layers=self.predictor_config.num_layers,
            embed_dropout=self.predictor_config.embed_dropout,
            rnn_dropout=self.predictor_config.rnn_dropout,
            rnn_type=self.rnn_type,
            pad=self.pad_id,
        )

        self.criterion = RNNTLoss(blank=self.blank_id, reduction="mean")

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, text: torch.Tensor,
                text_lengths: torch.Tensor):
        encoder_outputs, output_lengths = self.encoder(speech, speech_lengths)

        ys_in_pad = add_blank(text, self.blank_id, self.pad_id)
        #  [B, max_text_len + 1]  <pad> -> <blank>

        predictor_out, hidden_states = self.predictor(ys_in_pad)

        joint_out = self.joint(encoder_outputs, predictor_out)

        rnnt_text = text.to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        output_lengths = output_lengths.to(torch.int32)
        loss = torchaudio.functional.rnnt_loss(joint_out,
                                               rnnt_text,
                                               output_lengths,
                                               rnnt_text_lengths,
                                               blank=self.blank_id,
                                               reduction="mean")
        return loss

    @torch.no_grad()
    def recognize(self):
        pass
