import torchaudio.compliance.kaldi as kaldi
import torchaudio


class AudioFeatureExtractor:
    def __init__(self, feature_type="mfcc", sample_rate=16000,
                 frame_length=25.0, frame_shift=10.0,
                 num_mel_bins=80, dither=0.0, num_ceps=40,
                 high_freq=0, low_freq=20.0, use_torchaudio=False):
        assert feature_type in ["mfcc", "fbank", "spectrogram"], "feature_types not found"
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        self.num_mel_bins = num_mel_bins
        self.dither = dither
        self.num_ceps = num_ceps
        self.high_freq = high_freq
        self.low_freq = low_freq

        self.extractor = None
        if use_torchaudio:
            """
            if use_torchaudio is True, return features of dimension (channels, feature_dim, time)
            else return features of dimension (time, feature_dim)
            """
            if self.feature_type == "mfcc":
                self.extractor = torchaudio.transforms.MFCC(sample_rate=self.sample_rate, n_mfcc=self.num_mel_bins)
            elif self.feature_type == "fbank":
                self.extractor = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate,
                                                                      n_mels=self.num_mel_bins)
            elif self.feature_type == "spectrogram":
                self.extractor = torchaudio.transforms.Spectrogram()

    def __call__(self, waveform):
        """
        Args:
            waveform: Tensor of audio of dimension (channels, time)

        Returns:
            features: Tensor of audio features of dimension (channels, feature_dim, time)
        """
        waveform = waveform * (1 << 15)

        # use_torchaudio
        if self.extractor is not None:
            features = self.extractor(waveform)
            return features

        # use kaldi
        if self.feature_type == "fbank":
            features = kaldi.fbank(waveform,
                                   num_mel_bins=self.num_mel_bins,
                                   frame_length=self.frame_length,
                                   frame_shift=self.frame_shift,
                                   dither=self.dither,
                                   energy_floor=0.0,
                                   sample_frequency=self.sample_rate)
        elif self.feature_type == "mfcc":
            features = kaldi.mfcc(waveform,
                                  num_mel_bins=self.num_mel_bins,
                                  frame_length=self.frame_length,
                                  frame_shift=self.frame_shift,
                                  dither=self.dither,
                                  num_ceps=self.num_ceps,
                                  high_freq=self.high_freq,
                                  low_freq=self.low_freq,
                                  sample_frequency=self.sample_rate)
        else:
            features = kaldi.spectrogram(waveform,
                                         frame_length=self.frame_length,
                                         frame_shift=self.frame_shift,
                                         dither=self.dither,
                                         energy_floor=0.0,
                                         sample_frequency=self.sample_rate)
        if len(features.shape) == 2:
            features = features.transpose(0, 1).unsqueeze(0)
            # (time, feature_dim) -> (1, feature_dim, time)
        return features
