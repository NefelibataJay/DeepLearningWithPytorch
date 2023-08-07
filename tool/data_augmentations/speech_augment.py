import torch
import torchaudio
import random


class AudioAugmentation:
    def __init__(self, spec_aug=False, noise_aug=False, spec_trim=False):
        self.spec_aug = spec_aug
        self.noise_aug = noise_aug
        self.spec_trim = spec_trim


class SpecAugment:
    def __init__(self, max_t_mask=50, max_f_mask=10, num_t_mask=1, num_f_mask=1):
        # TODO add time warp
        # NOTE: if use torchaudio.transforms, the feature must be (1, feature_dim, time)
        # self.time_stretch = torchaudio.transforms.TimeStretch()
        self.timeMasking = torchaudio.transforms.TimeMasking(time_mask_param=max_t_mask)
        self.frequencyMasking = torchaudio.transforms.FrequencyMasking(freq_mask_param=max_f_mask)
        self.max_t_mask = max_t_mask
        self.max_f_mask = max_f_mask

        self.num_t_mask = num_t_mask
        self.num_f_mask = num_f_mask

    def __call__(self, feature):
        for i in range(self.num_t_mask):
            feature = self.timeMasking(feature)
        for i in range(self.num_f_mask):
            feature = self.frequencyMasking(feature)
        return feature

    def spec_aug(self, y):
        """
        y:   (1, feature_dim, time) -> (time, feature_dim)
        """
        y = y.squeeze(0).transpose(0, 1)
        max_frames = y.size(0)
        max_freq = y.size(1)
        # time mask
        for i in range(self.num_t_mask):
            start = random.randint(0, max_frames - 1)
            length = random.randint(1, self.max_t_mask)
            end = min(max_frames, start + length)
            y[start:end, :] = 0
        # freq mask
        for i in range(self.num_f_mask):
            start = random.randint(0, max_freq - 1)
            length = random.randint(1, self.max_f_mask)
            end = min(max_freq, start + length)
            y[:, start:end] = 0

        y = y.transpose(0, 1).unsqueeze(0)  # (time, feature_dim) -> (1, feature_dim, time)
        return y

    def time_warp(self, x: torch.Tensor, window: int = 80, mode: str = "bicubic"):
        """Time warping using torch.interpolate.

        Args:
            x: (Batch, Time, Freq)
            window: time warp parameter
            mode: Interpolate mode
        """

        # bicubic supports 4D or more dimension tensor
        org_size = x.size()
        if x.dim() == 3:
            # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
            x = x[:, None]

        t = x.shape[2]
        if t - window <= window:
            return x.view(*org_size)

        center = torch.randint(window, t - window, (1,))[0]
        warped = torch.randint(center - window, center + window, (1,))[0] + 1

        # left: (Batch, Channel, warped, Freq)
        # right: (Batch, Channel, time - warped, Freq)
        left = torch.nn.functional.interpolate(
            x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
        )
        right = torch.nn.functional.interpolate(
            x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False
        )

        if x.requires_grad:
            x = torch.cat([left, right], dim=-2)
        else:
            x[:, :, :warped] = left
            x[:, :, warped:] = right

        return x.view(*org_size)

    def spec_sub(self, feature, max_t=20, num_t_sub=3):
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

    def spec_trim(self, feature, max_trim=10):
        """ Trim tailing frames. Inplace operation.
            ref: TrimTail [https://arxiv.org/abs/2211.00522]
            feature [frames, d]
        """
        max_frames = feature.size(0)
        length = random.randint(1, max_trim)
        if length < max_frames / 2:
            feature = feature.clone().detach()[:max_frames - length]
        return feature


class SpeedPerturb:
    def __init__(self, speeds=None):
        if speeds is None:
            self.speeds = [0.9, 1.0, 1.1]
        else:
            self.speeds = speeds

    def __call__(self, waveform, sample_rate):
        speed = random.choice(self.speeds)
        # TODO: add speed perturb
        if speed != 1.0:
            pass
        return waveform
