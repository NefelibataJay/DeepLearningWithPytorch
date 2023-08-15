import torch
from omegaconf import OmegaConf

from models import ConformerCTCAttention


def mask_finished_scores(score: torch.Tensor,
                         flag: torch.Tensor) -> torch.Tensor:
    """
    If a sequence is finished, we only allow one alive branch. This function
    aims to give one branch a zero score and the rest -inf score.

    Args:
        score (torch.Tensor): A real value array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size, beam_size).
    """
    beam_size = score.size(-1)
    zero_mask = torch.zeros_like(flag, dtype=torch.bool)
    if beam_size > 1:
        unfinished = torch.cat((zero_mask, flag.repeat([1, beam_size - 1])),
                               dim=1)
        finished = torch.cat((flag, zero_mask.repeat([1, beam_size - 1])),
                             dim=1)
    else:
        unfinished = zero_mask
        finished = flag
    score.masked_fill_(unfinished, -float('inf'))
    score.masked_fill_(finished, 0)
    return score


def subsequent_mask(
        size: int,
        device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size).

    This mask is used only in decoder which works in an auto-regressive mode.
    This means the current step could only do attention with its left steps.

    In encoder, fully attention is used when streaming is not necessary and
    the sequence is not long. In this  case, no attention mask is needed.

    When streaming is need, chunk-based attention is used in encoder. See
    subsequent_chunk_mask for the chunk-based attention mask.

    Args:
        size (int): size of mask
        str device (str): "cpu" or "cuda" or torch.Tensor.device
        dtype (torch.device): result dtype

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_mask(3)
        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    arange = torch.arange(size, device=device)
    mask = arange.expand(size, size)
    arange = arange.unsqueeze(-1)
    mask = mask <= arange
    return mask


def mask_finished_preds(pred: torch.Tensor, flag: torch.Tensor,
                        eos: int) -> torch.Tensor:
    """
    If a sequence is finished, all of its branch should be <eos>

    Args:
        pred (torch.Tensor): A int array with shape
            (batch_size * beam_size, beam_size).
        flag (torch.Tensor): A bool array with shape
            (batch_size * beam_size, 1).

    Returns:
        torch.Tensor: (batch_size * beam_size).
    """
    beam_size = pred.size(-1)
    finished = flag.repeat([1, beam_size])
    return pred.masked_fill_(finished, eos)


def recognize(
        model: torch.nn.Module,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        beam_size: int = 10,
):
    """ Apply beam search on attention decoder

    Args:
        model (torch.nn.Module):
        speech (torch.Tensor): (batch, max_len, feat_dim)
        speech_lengths (torch.Tensor): (batch, )
        beam_size (int): beam size for beam search

    Returns:
        torch.Tensor: decoding result, (batch, max_result_len)
    """
    device = speech.device
    batch_size = speech.shape[0]
    # encoder   # (B, max_len, encoder_dim)
    encoder_out, encoder_out_len = model.encoder(speech, speech_lengths)
    max_len = encoder_out.size(1)
    encoder_dim = encoder_out.size(2)
    running_size = batch_size * beam_size
    # (B*N, max_len, encoder_dim)
    encoder_out = encoder_out.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, max_len, encoder_dim)
    # (B*N, 1, max_len)
    # encoder_mask = encoder_mask.unsqueeze(1).repeat(1, beam_size, 1, 1).view(running_size, 1, max_len)

    hyps = torch.ones([running_size, 1], dtype=torch.long, device=device).fill_(model.sos_id)  # (B*N, 1)
    scores = torch.tensor([0.0] + [-float('inf')] * (beam_size - 1), dtype=torch.float)
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
        top_k_index = mask_finished_preds(top_k_index, end_flag, model.eos_id)

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
        best_k_index = base_k_index.view(-1) + offset_k_index.view(-1)  # (B*N)

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
        end_flag = torch.eq(hyps[:, -1], model.eos_id).view(-1, 1)

    # 3. Select best of best
    scores = scores.view(batch_size, beam_size)
    # TODO: length normalization
    best_scores, best_index = scores.max(dim=-1)
    best_hyps_index = best_index + torch.arange(
        batch_size, dtype=torch.long, device=device) * beam_size
    best_hyps = torch.index_select(hyps, dim=0, index=best_hyps_index)

    best_hyps = best_hyps[:, 1:]  # remove sos
    return best_hyps, best_scores


def get_config(config_path):
    cfg = OmegaConf.load(config_path)
    return cfg


if __name__ == "__main__":
    config_path = "../conf/conformer_ctc_attention.yaml"
    cfg = get_config(config_path)
    cfg.model.num_classes = 12  # 10 classes for test
    model = ConformerCTCAttention(cfg)

    speech = torch.randn(2, 100, 80)
    speech_lengths = torch.tensor([100,80])
    hyps, scores = recognize(model, speech, speech_lengths, beam_size=1)
    print(hyps)
    print(scores)
