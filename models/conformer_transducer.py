import torch
import torchaudio
from omegaconf import DictConfig
from torchaudio.transforms import RNNTLoss

from models.decoder.transducer import TransducerJoint, RnnPredictor
from models.encoder.conformer_encoder import ConformerEncoder
from tool.common import add_blank, add_sos
from tool.loss import CTC


class ConformerTransducer(torch.nn.Module):
    def __init__(self, configs: DictConfig) -> None:
        super(ConformerTransducer, self).__init__()
        self.configs = configs

        if self.configs.weight_conf is not None:
            self.transducer_weight = self.configs.weight_conf.transducer_weight
            self.ctc_weight = self.configs.weight_conf.ctc_weight

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

        if self.ctc_weight > 0:
            self.ctc_fc = torch.nn.Linear(self.encoder_configs.encoder_dim, self.num_classes, bias=False)
            self.ctc_criterion = CTC(blank_id=self.blank_id, reduction="mean")

        self.criterion = RNNTLoss(blank=self.blank_id, reduction="mean")

    def forward(self, speech: torch.Tensor, speech_lengths: torch.Tensor, text: torch.Tensor,
                text_lengths: torch.Tensor):
        result = dict()
        encoder_outputs, output_lengths = self.encoder(speech, speech_lengths)

        ys_in_pad = add_sos(text, sos=self.sos_id, pad=self.pad_i)
        #  [B, max_text_len + 1]

        predictor_out, hidden_states = self.predictor(ys_in_pad)

        joint_out = self.joint(encoder_outputs, predictor_out)  # [B, input_length, target_length, joint_dim]

        rnnt_text = torch.where(text == self.pad_id, self.blank_id, text).to(torch.int32)
        rnnt_text = rnnt_text.to(torch.int32)
        rnnt_text_lengths = text_lengths.to(torch.int32)
        output_lengths = output_lengths.to(torch.int32)

        transducer_loss = self.criterion(joint_out, rnnt_text, output_lengths, rnnt_text_lengths)

        result["transducer_loss"] = transducer_loss
        result["encoder_outputs"] = encoder_outputs
        result["output_lengths"] = output_lengths

        if self.ctc_weight > 0:
            logits = self.ctc_fc(encoder_outputs)
            ctc_loss = self.ctc_criterion(hs_pad=logits,
                                          ys_pad=text,
                                          h_lens=output_lengths,
                                          ys_lens=text_lengths, )
            ctc_loss = ctc_loss * self.ctc_weight
            loss = transducer_loss * self.transducer_weight + ctc_loss
            result["ctc_loss"] = ctc_loss
            result["loss"] = loss
            return loss, transducer_loss, ctc_loss

        result["loss"] = transducer_loss
        return transducer_loss
