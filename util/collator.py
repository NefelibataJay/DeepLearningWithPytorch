import torch


def asr_collate_fn(batch):
    """
    batch -> speech_feature, input_lengths, transcript, target_lengths
    Return:
        inputs : [batch, max_time, dim]
        input_lengths: [batch]
        targets: [batch, max_len]
        target_lengths: [batch]
    """
    inputs = [i[0] for i in batch]
    input_lengths = torch.IntTensor([i[1] for i in batch])
    targets = [torch.IntTensor(i[2]) for i in batch]
    target_lengths = torch.IntTensor([i[3] for i in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(dtype=torch.int)

    return inputs, input_lengths, targets, target_lengths


def accent_collate_fn(batch):
    """
    batch -> inputs, input_lengths, targets, target_lengths, accents_id, speakers_id
    Return:
        inputs : [batch, max_time, dim]
        input_lengths: [batch]
        targets: [batch, max_len]
        target_lengths: [batch]
        accents_id: [batch]
        speakers_id: [batch]
    """
    inputs = [i[0] for i in batch]
    input_lengths = torch.IntTensor([i[1] for i in batch])
    targets = [torch.IntTensor(i[2]) for i in batch]
    target_lengths = torch.IntTensor([i[3] for i in batch])
    accents_id = torch.IntTensor([i[4] for i in batch])
    speakers_id = torch.IntTensor([i[5] for i in batch])

    inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=0)
    targets = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=0).to(dtype=torch.int)

    return inputs, input_lengths, targets, target_lengths, accents_id, speakers_id
