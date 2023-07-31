import torchaudio.compliance.kaldi as kaldi


class FeatureExtractor:
    def __init__(self, feature_type="mfcc"):
        assert feature_type in ["mfcc", "fbank", "spectrogram"], "feature_types not found"
        self.feature_type = feature_type

    def __call__(self, waveform, sample_rate=16000,
                 frame_length=25.0, frame_shift=10.0,
                 num_mel_bins=23, dither=0.0, num_ceps=40,
                 high_freq=0, low_freq=20.0):
        waveform = waveform * (1 << 15)
        if self.feature_type == "spectrogram":
            mat = kaldi.spectrogram(waveform,
                                    frame_length=frame_length,
                                    frame_shift=frame_shift,
                                    dither=dither,
                                    energy_floor=0.0,
                                    sample_frequency=sample_rate)
        elif self.feature_type == "fbank":
            mat = kaldi.fbank(waveform,
                              num_mel_bins=num_mel_bins,
                              frame_length=frame_length,
                              frame_shift=frame_shift,
                              dither=dither,
                              energy_floor=0.0,
                              sample_frequency=sample_rate)
        elif self.feature_type == "mfcc":
            mat = kaldi.mfcc(waveform,
                             num_mel_bins=num_mel_bins,
                             frame_length=frame_length,
                             frame_shift=frame_shift,
                             dither=dither,
                             num_ceps=num_ceps,
                             high_freq=high_freq,
                             low_freq=low_freq,
                             sample_frequency=sample_rate)
