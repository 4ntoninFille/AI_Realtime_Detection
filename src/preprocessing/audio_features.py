from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class AudioFeatures:
    chroma_stft: float
    rms: float
    spectral_centroid: float
    spectral_bandwidth: float
    rolloff: float
    zero_crossing_rate: float
    mfcc: List[float]

    def to_array(self) -> np.ndarray:
        return np.array([
            self.chroma_stft,
            self.rms,
            self.spectral_centroid,
            self.spectral_bandwidth,
            self.rolloff,
            self.zero_crossing_rate,
            *self.mfcc
        ])

    @classmethod
    def from_array(cls, array: np.ndarray):
        return cls(
            chroma_stft=array[0],
            rms=array[1],
            spectral_centroid=array[2],
            spectral_bandwidth=array[3],
            rolloff=array[4],
            zero_crossing_rate=array[5],
            mfcc=array[6:].tolist()
        )
