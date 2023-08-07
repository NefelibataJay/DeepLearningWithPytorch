import random

import torch
import torchaudio
from espnet2.layers.time_warp import TimeWarp
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
from torchaudio.compliance import kaldi

from tool.data_augmentations.speech_augment import SpeedPerturb, SpecAugment
from tool.feature_extractor.speech_feature_extractor import AudioFeatureExtractor
import torchaudio.functional as F
from IPython.display import Audio
from torchaudio.utils import download_asset


def test_speaug():
    audio_path = "E:/Desktop/resources/test.wav"
    sample_rate = 16000

    waveform, sr = torchaudio.load(audio_path)  # (channel, time) (1, time)

    # Resample
    # if sr != sample_rate:
    #     waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    # Add noise
    # noisy_speeches = add_noise(waveform)

    # SpeedPerturb
    # speed_perturb = torchaudio.transforms.SpeedPerturbation(sample_rate, [0.9, 1.0, 1.1,1.0,1.0])
    # waveform, _ = speed_perturb(waveform)

    # extract_feature
    feature_extractor = AudioFeatureExtractor(feature_type="fbank", dither=0.1)
    features = feature_extractor(waveform)  # (1, feature_dim, time)
    plot_wav(features)

    # SpecAugment
    spec_aug = SpecAugment(max_t_mask=50, max_f_mask=10, num_t_mask=1, num_f_mask=1)
    features1 = spec_aug(features)
    plot_wav(features1)


def add_noise(speech):
    SAMPLE_NOISE = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-babb-mc01-stu-clo-8000hz.wav")
    noise, _ = torchaudio.load(SAMPLE_NOISE)
    noise = noise[:, : speech.shape[1]]
    snr_dbs = torch.tensor([20, 10, 3])
    noisy_speeches = F.add_noise(speech, noise, snr_dbs)
    return noisy_speeches


def plot_wav(waveform, title="nn"):
    waveform = waveform.squeeze(0)  # 使时间维度（帧数）在第一维度
    plt.imshow(waveform, aspect='auto', origin='lower')
    plt.xlabel('frame')
    plt.ylabel('dim')
    plt.title(title)
    plt.ste_xmax = 500
    plt.colorbar(format='%+2.0f dB')
    plt.show()
