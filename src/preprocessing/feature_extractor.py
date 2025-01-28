import librosa
import numpy as np
from typing import List
from .audio_features import AudioFeatures

class FeatureExtractor:
    """Class responsible for extracting features from audio data"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def extract_features(self, audio_block: np.ndarray) -> AudioFeatures:
        chroma = librosa.feature.chroma_stft(y=audio_block, sr=self.sample_rate)
        chroma_mean = np.mean(chroma)
        
        rms = librosa.feature.rms(y=audio_block)
        rms_mean = np.mean(rms)
        
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_block, sr=self.sample_rate)
        centroid_mean = np.mean(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_block, sr=self.sample_rate)
        bandwidth_mean = np.mean(spectral_bandwidth)
        
        rolloff = librosa.feature.spectral_rolloff(y=audio_block, sr=self.sample_rate)
        rolloff_mean = np.mean(rolloff)
        
        zero_crossing_rate = librosa.feature.zero_crossing_rate(audio_block)
        zcr_mean = np.mean(zero_crossing_rate)
        
        mfccs = librosa.feature.mfcc(y=audio_block, sr=self.sample_rate, n_mfcc=20)
        mfcc_means = [np.mean(mfcc) for mfcc in mfccs]

        return AudioFeatures(
            chroma_stft=chroma_mean,
            rms=rms_mean,
            spectral_centroid=centroid_mean,
            spectral_bandwidth=bandwidth_mean,
            rolloff=rolloff_mean,
            zero_crossing_rate=zcr_mean,
            mfcc=mfcc_means
        )
