import torch
import torch.nn.functional as F

from tool.search.hypotheses import BeamHypotheses


class BeamSearch:
    def __init__(self, length_penalty: int = 0, beam_size: int = 1, max_length: int = 100, sos_id: int = 1,
                 eos_id: int = 2, blank_id: int = 3, pad_id: int = 0):
        self.beam_size = beam_size
        self.max_length = max_length
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.blank_id = blank_id
        self.length_penalty = length_penalty

    def __call__(self, log_probs, output_lens, _type="ctc"):
        assert _type in ["ctc", "attention", "transducer"], "Decode_type Not Support!"
        if _type == "ctc":
            hyps, scores = self.ctc_beam_search(log_probs, output_lens)
        elif _type == "attention":
            # TODO : attention beam search
            pass
        elif _type == "transducer":
            # TODO : transducer beam search
            pass

        return hyps, scores

    def ctc_beam_search(self, log_probs: torch.Tensor, output_lens: torch.Tensor):
        batch_size = log_probs.shape[0]
        topk_prob, topk_index = log_probs.topk(self.beam_size, dim=2)
        hyps = None
        scores = None
        return hyps, scores


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
