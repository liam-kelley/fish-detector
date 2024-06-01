import librosa
import numpy as np
import matplotlib.pyplot as plt
import librosa.display
from typing import List


class Feature_Extractor:
    def __init__(self, max_duration_of_loaded_wav=60):
        """
        Can extract various features from wav files, returns them as a dict.
        
        Use the extract_features method to extract features.
        Currently only returns :
            pcen.
        
        Can also return if uncommented :
            mel, logmel
            mfcc, delta_mfcc, mel_un_normalized
            rms, spectral_centroid, spectral_bandwidth, spectral_contrast,
            spectral_flatness, spectral_bandwidth, spectral_rolloff,
            poly_features, zero_crossing_rate
        """

        self.sr = 22050
        self.n_fft = 1024
        self.hop = 256
        self.n_mels = 128
        self.n_mfcc = 32
        self.fmax = 11025
        
        # Added to control length of the long files
        self.max_duration_of_loaded_wav = max_duration_of_loaded_wav

    def norm(self, y):
        return y / np.max(np.abs(y))

    def mel(self, y):
        assert np.max(y) <= 1, np.max(y)
        mel_spec = librosa.feature.melspectrogram(
            y=y,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        # mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def logmel(self, mel_spec):
        mel_spec = np.log(mel_spec + 1e-8)
        mel_spec = mel_spec.astype(np.float32)
        return mel_spec

    def pcen(self, y):
        assert np.max(y) <= 1
        mel_spec = librosa.feature.melspectrogram(
            y=y * (2**32),
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
            n_mels=self.n_mels,
            fmax=self.fmax,
        )
        pcen = librosa.core.pcen(mel_spec, sr=self.sr)
        pcen = pcen.astype(np.float32)
        return pcen

    def rms(self, S):
        return librosa.feature.rms(S=S, frame_length=self.n_fft)

    def mfcc(self, mel):
        return librosa.feature.mfcc(S=mel)

    def spectral_centroid(self, S):
        return librosa.feature.spectral_centroid(S=S, sr=self.sr)

    def spectral_bandwidth(self, S):
        return librosa.feature.spectral_bandwidth(S=S, sr=self.sr)

    def spectral_contrast(self, S, n_bands=6):
        return librosa.feature.spectral_contrast(S=S, sr=self.sr, n_bands=n_bands)

    def spectral_flatness(self, S):
        return librosa.feature.spectral_flatness(S=S)

    def spectral_bandwidth(self, S):
        return librosa.feature.spectral_bandwidth(S=S, sr=self.sr)

    def spectral_rolloff(self, S, roll_percent=0.9):
        return librosa.feature.spectral_rolloff(
            S=S, sr=self.sr, roll_percent=roll_percent
        )

    def poly_features(self, S, order=1):
        return librosa.feature.poly_features(S=S, sr=self.sr, order=order)

    def zero_crossing_rate(self, y):
        assert np.max(y) <= 1
        return librosa.feature.zero_crossing_rate(
            y, frame_length=self.n_fft, hop_length=self.hop
        )

    def delta_mfcc(self, mfcc, order=1, width=9):
        return librosa.feature.delta(mfcc, order=order, width=width)

    def draw_spec(self, matrix, save_path=None, name="Spectrogram"):
        '''
        Use after extracting a spectrogram-like feature.
        
        result = fe.extract_features(..., features_to_extract=['pcen'])
        fe.draw_spec(result['pcen'])
        '''
        plt.imshow(matrix, aspect="auto", interpolation="none")
        plt.title(name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def draw_plot(self, line, save_path=None, name="temp"):
        plt.plot(line)
        plt.title(name)
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
        plt.close()

    def visualize_result(self, result):
        for k in result.keys():
            if result[k].shape[1] == 1:
                self.draw_plot(result[k], name=k)
            elif result[k].shape[1] > 1:
                self.draw_spec(result[k], name=k)

    def extract_features(self, audio_path : str, features_to_extract : List[str]= ['pcen']) -> dict:
        result = {}
        
        # Load Audio
        result["waveform"], _ = librosa.load(audio_path, sr=self.sr, duration=self.max_duration_of_loaded_wav)
        # Normalize audio to have max 1.0
        result["waveform"] = self.norm(result["waveform"])

        # In the context of automatic speech recognition and acoustic event detection, an
        # adaptive procedure named per-channel energy normalization (PCEN) has recently 
        # shown to outperform the pointwise logarithm of mel-frequency spectrogram (logmelspec)
        # PCEN is pretty much compressing time-invariant noises to a constant low-level
        #   + compressing foreground noise to a constant higher level.
        # A time threshold differentiates between long noises and short noises
        if "pcen" in features_to_extract:
            result["pcen"] = self.pcen(result["waveform"]).astype(np.float32)
        
        other_possible_features = ["mel", "logmel",
                "mfcc", "delta_mfcc", "mel_un_normalized",
                "rms", "spectral_centroid", "spectral_bandwidth",
                "spectral_contrast", "spectral_flatness",
                "spectral_bandwidth", "spectral_rolloff",
                "poly_features", "zero_crossing_rate"]
        
        if any(feature in features_to_extract for feature in other_possible_features):
            raise NotImplementedError, "Please modify the feature extractor script to extract these features. The code is probably just commented."
            # Other possible features
            # result["mel"] = self.mel(result["waveform"]).astype(np.float32)
            # result["logmel"] = self.logmel(result["mel"]).astype(np.float32)
            # result["mfcc"] = self.mfcc(result["logmel"]).astype(np.float32)
            # result["delta_mfcc"] = self.delta_mfcc(result["mfcc"], order=1, width=9).astype(np.float32)
            # result["mel_un_normalized"] = result["mel"]
            
            # More possible features
            # result["spec"], result["phase"] = librosa.magphase(
            #     librosa.stft(result["waveform"], hop_length=256, n_fft=1024)
            # )
            # del result["phase"]
            # result["rms"] = self.rms(result["spec"]).astype(np.float32)
            # result["spectral_centroid"] = self.spectral_centroid(result["spec"]).astype(np.float32)
            # result["spectral_bandwidth"] = self.spectral_bandwidth(result["spec"]).astype(np.float32)
            # result["spectral_contrast"] = self.spectral_contrast(result["spec"], n_bands=6).astype(np.float32)
            # result["spectral_flatness"] = self.spectral_flatness(result["spec"]).astype(np.float32)
            # result["spectral_bandwidth"] = self.spectral_bandwidth(result["spec"]).astype(np.float32)
            # result["spectral_rolloff"] = self.spectral_rolloff(result["spec"]).astype(np.float32)
            # result["poly_features"] = self.poly_features(result["spec"], order=1).astype(np.float32)
            # result["zero_crossing_rate"] = self.zero_crossing_rate(result["waveform"]).astype(np.float32)

        del result["waveform"]

        # Transpose features
        for k in result.keys():
            result[k] = result[k].T

        return result