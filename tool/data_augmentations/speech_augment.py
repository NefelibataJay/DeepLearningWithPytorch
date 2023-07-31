import torchaudio
import torchaudio.transforms as T
import random


class SpecAugment:
    def __init__(self, max_t_mask=10, max_f_mask=10, num_t_mask=1, num_f_mask=1):
        # TODO add time warp
        self.timeMasking = torchaudio.transforms.TimeMasking(time_mask_param=max_t_mask)
        self.frequencyMasking = torchaudio.transforms.FrequencyMasking(freq_mask_param=max_f_mask)
        self.max_t_mask = max_t_mask
        self.num_f_mask = max_f_mask
        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask

    def __call__(self, feature):
        for i in range(self.num_t_mask):
            feature = self.timeMasking(feature)
        for i in range(self.num_t_mask):
            feature = self.frequencyMasking(feature)
        return feature


def spec_sub(feature, max_t=20, num_t_sub=3):
    y = feature.clone().detach()
    max_frames = y.size(0)
    for i in range(num_t_sub):
        start = random.randint(0, max_frames - 1)
        length = random.randint(1, max_t)
        end = min(max_frames, start + length)
        # only substitute the earlier time chosen randomly for current time
        pos = random.randint(0, start)
        y[start:end, :] = feature[start - pos:end - pos, :]
    return y


def spec_trim(feature, max_trim=10):
    """ Trim tailing frames. Inplace operation.
        ref: TrimTail [https://arxiv.org/abs/2211.00522]
        feature [frames, d]
    """
    max_frames = feature.size(0)
    length = random.randint(1, max_trim)
    if length < max_frames / 2:
        feature = feature.clone().detach()[:max_frames - length]
    return feature


def speed_perturb(waveform, sample_rate, speeds=None):
    if speeds is None:
        speeds = [0.9, 1.0, 1.1]
    speed = random.choice(speeds)
    if speed != 1.0:
        wav, _ = torchaudio.sox_effects.apply_effects_tensor(
            waveform, sample_rate,
            [['speed', str(speed)], ['rate', str(sample_rate)]])
        waveform = wav
    return waveform
