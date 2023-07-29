import torch

from tool.common import remove_duplicates_and_blank
from models.modules.mask import *


class GreedySearch:
    def __init__(self,decode_type: str, max_length: int = 128, sos_id: int = 1, eos_id: int = 2, blank_id: int = 3, pad_id: int = 0):
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id
        self.model_type = decode_type

    def __call__(self, model, inputs, input_lengths):
        if self.model_type == "ctc":
            return self.ctc_greedy_search(inputs, input_lengths)

    def _greedy_search(self, log_probs):
        """
        TODO add Batch
        Given a sequence, get the best path
            Args:
                log_probs (Tensor): Logit tensors. Shape `[num_seq, num_label]`.
            Returns:
                List[int]: label Token
        """
        indices = torch.argmax(log_probs, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank_id]
        return indices

    def ctc_greedy_search(self, logits, encoder_out_lens):
        """ implement ctc greedy search from wenet """
        logits = logits.log_softmax(dim=-1)
        batch_size = logits.shape[0]
        max_len = logits.shape[1]
        ctc_probs = logits.log_softmax(dim=2)
        # topk_index = log_probs.argmax(2)
        topk_prob, topk_index = ctc_probs.topk(1, dim=2)  # (B, max_len, 1)
        topk_index = topk_index.view(batch_size, max_len)  # (B, max_len)
        mask = make_pad_mask(encoder_out_lens, maxlen=max_len)  # (B, max_len)
        topk_index = topk_index.masked_fill_(mask, self.eos_id)  # (B, max_len)
        hyps = [hyp.tolist() for hyp in topk_index]
        scores = topk_prob.max(1)
        hyps = [remove_duplicates_and_blank(hyp, self.blank_id) for hyp in hyps]
        return hyps, scores

    def rnnt_single_greedy_search(self, transducer, encoder_output, output_length, n_steps: int = 64):
        """
        Note: only support batch_size = 1
            对encoder 的每个时间步进行解码，直到遇到blank时代表这句话**可能**解码完成
        Args:
            transducer : Transducer Model
            encoder_output : (batch_size, input_length, encoder_output_size)
            output_length : (batch_size, )
        """
        # NOTE only support batch_size = 1
        hyps = list()
        batch_size = encoder_output.size(0)  # must be 1
        pred_input_step = encoder_output.new_zeros(batch_size, 1).fill_(self.sos_id).long()  # [[sos_id]]
        # first decode input is sos
        t = 0
        prev_out_nblk = True
        pred_out_step = None
        per_frame_max_noblk = 64
        per_frame_noblk = 0
        while t < output_length:
            encoder_out_step = encoder_output[:, t:t + 1, :]
            if prev_out_nblk:
                pred_out_step, hidden_states = transducer.predictor(pred_input_step)
                # decoder_outputs (batch_size, 1, decoder_output_size)

            joint_out_step = transducer.joint(encoder_out_step, pred_out_step)
            joint_out_probs = joint_out_step.log_softmax(dim=-1)

            joint_out_max = joint_out_probs.argmax(dim=-1).squeeze()

            if joint_out_max != self.blank_id:
                # decoding not blank , add to hyps
                hyps.append(joint_out_max.item())
                prev_out_nblk = True

                # 对于RNN而言，当前的时间步解码出的不是blank，那么下一个时间步的输入就是当前时间步的输出
                # 根据当前时间步的结果继续解码
                per_frame_noblk = per_frame_noblk + 1
                pred_input_step = joint_out_max.reshape(1, 1)
            if joint_out_max == self.blank_id or per_frame_noblk >= per_frame_max_noblk:
                if joint_out_max == self.blank_id:
                    # 解码出blank，说明这句话可能解码完成
                    prev_out_nblk = False
                # next time step
                t = t + 1
                per_frame_noblk = 0
        return hyps

    def rnnt_greedy_search(self, transducer, encoder_outputs, output_lengths):
        """TODO add Batch
        思路: 上面的单条解码

        Args:
            transducer : Transducer Model
            encoder_output : (batch_size, input_length, encoder_output_size)
            output_length : (batch_size, )
        """
        hyps = list()
        pass

    def attention_greedy_search(self, encoder_outputs, encoder_output_lengths, decoder):
        pass


def transformer_greedy_search(decoder, encoder_outputs, encoder_output_lengths):
    beam_size = 1
    sos = 1
    eos = 2
    pad = 0
    batch_size = encoder_outputs.size(0)
    hidden_size = encoder_outputs.size(2)
    vocab_size = decoder.num_classes
    max_length = 128

    input_var = encoder_outputs.new_zeros(batch_size, 1).long()
    input_var = input_var.fill_(sos)  # add sos

    for di in range(1, max_length):
        input_lengths = torch.IntTensor(batch_size).fill_(di)
        dec_outputs, _, _ = decoder.forward_step(
            decoder_inputs=input_var,
            decoder_input_lengths=input_lengths,
            encoder_outputs=encoder_outputs,
            encoder_output_lengths=encoder_output_lengths,
            positional_encoding_length=di, )
        # dec_outputs (batch_size, max_length, vocab_size)
        topk_prob, topk_index = decoder.fc(dec_outputs).log_softmax(dim=-1).topk(beam_size, dim=-1)
        # topk_index = decoder.fc(dec_outputs).log_softmax(dim=-1).argmax(dim=-1)
        # topk_index is token_id
        new_token_id = topk_index[:, -1, :]
        input_var = torch.cat([input_var, new_token_id], dim=1)
        if torch.all(new_token_id < eos + 1):
            break
    return input_var


def rnnt_greedy_search():
    pass
