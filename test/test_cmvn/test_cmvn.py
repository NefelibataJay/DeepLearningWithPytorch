import json
import math

import numpy as np
import torch
import torchaudio

from tool.feature_extractor.speech_feature_extractor import AudioFeatureExtractor


def _load_json_cmvn(json_cmvn_file):
    """ Load the json format cmvn stats file and calculate cmvn

    Args:
        json_cmvn_file: cmvn stats file in json format

    Returns:
        a numpy array of [means, vars]
    """
    with open(json_cmvn_file) as f:
        cmvn_stats = json.load(f)

    means = cmvn_stats['mean_stat']
    variance = cmvn_stats['var_stat']
    count = cmvn_stats['frame_num']
    for i in range(len(means)):
        means[i] /= count
        variance[i] = variance[i] / count - means[i] * means[i]
        if variance[i] < 1.0e-20:
            variance[i] = 1.0e-20
        variance[i] = 1.0 / math.sqrt(variance[i])
    cmvn = np.array([means, variance])
    return cmvn


class GlobalCMVN(torch.nn.Module):
    def __init__(self,
                 mean: torch.Tensor,
                 istd: torch.Tensor,
                 norm_var: bool = True):
        """
        Args:
            mean (torch.Tensor): mean stats
            istd (torch.Tensor): inverse std, std which is 1.0 / std
        """
        super().__init__()
        assert mean.shape == istd.shape
        self.norm_var = norm_var
        # The buffer can be accessed from this module using self.mean
        self.register_buffer("mean", mean)
        self.register_buffer("istd", istd)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (batch, max_len, feat_dim)

        Returns:
            (torch.Tensor): normalized feature
        """
        x = x - self.mean
        if self.norm_var:
            x = x * self.istd
        return x


# if __name__ == "__main__":
def test_cmvn():
    audio_path = "E:/Desktop/resources/test.wav"
    waveform, sr = torchaudio.load(audio_path)
    feature_extractor = AudioFeatureExtractor(feature_type="fbank", dither=0.1)
    features = feature_extractor(waveform).transpose(1, 2)  # (1, time, feature_dim)

    mean, istd = _load_json_cmvn('./global_cmvn')
    global_cmvn = GlobalCMVN(torch.from_numpy(mean).float(), torch.from_numpy(istd).float())

    xs = global_cmvn(features)

