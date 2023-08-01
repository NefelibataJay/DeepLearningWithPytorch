import torch
import torch.nn.functional as F

from models.modules.mask import subsequent_mask, mask_finished_scores, mask_finished_preds
from tool.search.hypotheses import BeamHypotheses


class BeamSearch:
    def __init__(self, length_penalty: int = 0, beam_size: int = 10, max_length: int = 100, sos_id: int = 1,
                 eos_id: int = 2, blank_id: int = 3, pad_id: int = 0):
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id
        self.length_penalty = length_penalty

    def ctc_beam_search(self, log_probs: torch.Tensor, output_lens: torch.Tensor):
        """ TODO """
        batch_size = log_probs.shape[0]
        topk_prob, topk_index = log_probs.topk(self.beam_size, dim=2)
        hyps = None
        scores = None
        return hyps, scores

    def ctc_prefix_beam_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            model,
    ):
        batch_size = speech.shape[0]
        beam_size = self.beam_size

        encoder_out, encoder_out_len = model.encoder()

        assert batch_size == 1

    def attention_beam_search(
            self,
            speech: torch.Tensor,
            speech_lengths: torch.Tensor,
            model,
    ):
        device = speech.device
        batch_size = speech.shape[0]
        beam_size = self.beam_size
        # encoder   # (B, max_len, encoder_dim)
        encoder_out, encoder_out_len = model.encoder(speech, speech_lengths)
        max_len = encoder_out.size(1)
        encoder_dim = encoder_out.size(2)
        running_size = batch_size * beam_size
        # (B*N, max_len, encoder_dim)
        encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, max_len, encoder_dim)
        # (B*N, 1, max_len)
        # encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, 1, max_len)

        hyps = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(self.sos_id)  # (B*N, 1)
        scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1),dtype=torch.float)
        scores = scores.repeat([batch_size]).unsqueeze(1).to(device)
        end_flag = torch.zeros_like(scores, dtype=torch.bool, device=device)

        # decoder step by step
        for i in range(1, max_len + 1):
            # Stop if all batch and all beam produce eos
            if end_flag.sum() == running_size:
                break
            # 2.1 Forward decoder step
            hyps_mask = subsequent_mask(i).unsqueeze(0).repeat(running_size, 1, 1).to(device)  # (B*N, i, i)

            # logp: (B*N, vocab)
            logp = model.decoder.forward_one_step(encoder_out, encoder_out_len, hyps, hyps_mask)
            # 2.2 First beam prune: select topk the best prob at current time
            top_k_logp, top_k_index = logp.topk(beam_size)  # (B*N, N)

            top_k_logp = mask_finished_scores(top_k_logp, end_flag)
            top_k_index = mask_finished_preds(top_k_index, end_flag, self.eos_id)

            # 2.3 Second beam prune: select topk score with history
            scores = scores + top_k_logp  # (B*N, N), broadcast add
            scores = scores.view(batch_size, beam_size * beam_size)  # (B, N*N)
            scores, offset_k_index = scores.topk(k=beam_size)  # (B, N)

            scores = scores.view(-1, 1)  # (B*N, 1)
            # 2.4. Compute base index in top_k_index,
            # regard top_k_index as (B*N*N),regard offset_k_index as (B*N),
            # then find offset_k_index in top_k_index
            base_k_index = torch.arange(batch_size, device=device).view(
                -1, 1).repeat([1, beam_size])  # (B, N)
            base_k_index = base_k_index * beam_size * beam_size
            best_k_index = base_k_index.view(-1) + offset_k_index.view(
                -1)  # (B*N)

            # 2.5 Update best hyps
            best_k_pred = torch.index_select(top_k_index.view(-1),
                                             dim=-1,
                                             index=best_k_index)  # (B*N)
            best_hyps_index = best_k_index // beam_size
            last_best_k_hyps = torch.index_select(
                hyps, dim=0, index=best_hyps_index)  # (B*N, i)
            hyps = torch.cat((last_best_k_hyps, best_k_pred.view(-1, 1)),
                             dim=1)  # (B*N, i+1)

            # 2.6 Update end flag
            end_flag = torch.eq(hyps[:, -1], self.eos_id).view(-1, 1)

        # 3. Select best of best
        scores = scores.view(batch_size, beam_size)
        # TODO: length normalization
        best_scores, best_index = scores.max(dim=-1)
        best_hyps_index = best_index + torch.arange(
            batch_size, dtype=torch.long, device=device) * beam_size
        best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)

        best_hyps = best_hyps[:, 1:] # remove sos
        return best_hyps, best_scores


def transformer_beam_search(decoder, beam_size, encoder_outputs):
    beam_size = 1
    sos = 1
    eos = 2
    pad = 0
    k = 1  # beam_size
    batch_size = encoder_outputs.size(0)
    hidden_size = encoder_outputs.size(2)
    vocab_size = decoder.num_classes
    max_length = 128

    k_prev_words = torch.full((k, batch_size), sos, dtype=torch.long)  # (k, 1)
    # 此时输出序列中只有sos token
    seqs = k_prev_words  # (k, batch_size)
    # 初始化scores向量为0
    top_k_scores = torch.zeros(k, batch_size)
    complete_seqs = list()
    complete_seqs_scores = list()
    step = 1
    hidden = torch.zeros(batch_size, k, hidden_size)  # h_0: (batch_size, k, hidden_size)
    while True:
        outputs, hidden = decoder(k_prev_words, hidden)  # outputs: (k, seq_len, vocab_size)
        next_token_logits = outputs[:, -1, :]  # (k, vocab_size)
        if step == 1:
            # 因为最开始解码的时候只有一个结点<sos>,所以只需要取其中一个结点计算topk
            top_k_scores, top_k_words = next_token_logits[0].topk(k, dim=0, largest=True, sorted=True)
        else:
            # 此时要先展开再计算topk，如上图所示。
            # top_k_scores: (k) top_k_words: (k)
            top_k_scores, top_k_words = next_token_logits.view(-1).topk(k, 0, True, True)
        prev_word_inds = top_k_words / vocab_size  # (k)  实际是beam_id
        next_word_inds = top_k_words % vocab_size  # (k)  实际是token_id
        # seqs: (k, step) ==> (k, step+1)
        seqs = torch.cat([seqs[prev_word_inds], next_word_inds.unsqueeze(1)], dim=1)

        # 当前输出的单词不是eos的有哪些(输出其在next_wod_inds中的位置, 实际是beam_id)
        incomplete_inds = [ind for ind, next_word in enumerate(next_word_inds) if
                           next_word != eos]
        # 输出已经遇到eos的句子的beam id(即seqs中的句子索引)
        complete_inds = list(set(range(len(next_word_inds))) - set(incomplete_inds))

        if len(complete_inds) > 0:
            complete_seqs.extend(seqs[complete_inds].tolist())  # 加入句子
            complete_seqs_scores.extend(top_k_scores[complete_inds])  # 加入句子对应的累加log_prob
        # 减掉已经完成的句子的数量，更新k, 下次就不用执行那么多topk了，因为若干句子已经被解码出来了
        k -= len(complete_inds)

        if k == 0:  # 完成
            break

        # 更新下一次迭代数据, 仅专注于那些还没完成的句子
        seqs = seqs[incomplete_inds]
        hidden = hidden[prev_word_inds[incomplete_inds]]
        top_k_scores = top_k_scores[incomplete_inds].unsqueeze(1)  # (s, 1) s < k
        k_prev_words = next_word_inds[incomplete_inds].unsqueeze(1)  # (s, 1) s < k

        if step > max_length:  # decode太长后，直接break掉
            break
        step += 1
    i = complete_seqs_scores.index(max(complete_seqs_scores))  # 寻找score最大的序列
    # 有些许问题，在训练初期一直碰不到eos时，此时complete_seqs为空
    seq = complete_seqs[i]

    return seq


if __name__ == '__main__':
    pass
