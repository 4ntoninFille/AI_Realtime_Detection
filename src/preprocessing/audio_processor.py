import logging
import os
from typing import List, Optional
import numpy as np
import soundfile as sf
from .feature_extractor import FeatureExtractor
from .audio_features import AudioFeatures

logger = logging.getLogger(__name__)

audiofeatures = [('chroma_stft', 'float'), 
        ('rms', 'float'),
        ('spectral_centroid', 'float'),
        ('spectral_bandwidth', 'float'),
        ('rolloff', 'float'),
        ('zero_crossing_rate', 'float'),
        ('mfcc1', 'float'),
        ('mfcc2', 'float'),
        ('mfcc3', 'float'),
        ('mfcc4', 'float'),
        ('mfcc5', 'float'),
        ('mfcc6', 'float'),
        ('mfcc7', 'float'),
        ('mfcc8', 'float'),
        ('mfcc9', 'float'),
        ('mfcc10', 'float'),
        ('mfcc11', 'float'),
        ('mfcc12', 'float'),
        ('mfcc13', 'float'),
        ('mfcc14', 'float'),
        ('mfcc15', 'float'),
        ('mfcc16', 'float'),
        ('mfcc17', 'float'),
        ('mfcc18', 'float'),
        ('mfcc19', 'float'),
        ('mfcc20', 'float'),
        ('label', 'S10')]

class AudioProcessor:
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate=sample_rate)

    def _load_audio_file(self, path: str) -> List[np.ndarray]:
        try:
            data, file_sample_rate = sf.read(path)

            if len(data.shape) > 1:
                data = np.mean(data, axis=1)
            
            if file_sample_rate != self.sample_rate:
                logger.warning(f"Warning: File sample rate ({file_sample_rate}) differs from expected ({self.sample_rate})")
                self.sample_rate = file_sample_rate
                self.feature_extractor.sample_rate = file_sample_rate;
            
            data = data.astype(np.float32)
            if np.abs(data).max() > 1.0:
                data = data / np.abs(data).max()
            
            # Split into 1 second blocks
            block_size = self.sample_rate
            n_blocks = len(data) // block_size
            
            return [data[i*block_size:(i+1)*block_size] for i in range(n_blocks)]

        except Exception as e:
            raise ValueError(f"Error loading audio file: {str(e)}")

    def load_custome_audio(self, path: str) -> np.ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Audio file not found: {path}")
        
        try:
            audio_blocks = self._load_audio_file(path)
        except ValueError as e:
            raise ValueError(f"Error loading audio file: {str(e)}")
        
        features = self._process_audio_blocks(audio_blocks)
        
        return np.array([feature.to_array() for feature in features])

    def _process_audio_blocks(self, audio_blocks: List[np.ndarray]) -> List[AudioFeatures]:
        return [self.feature_extractor.extract_features(block) for block in audio_blocks]

    def process_realtime_block(self, audio_block: np.ndarray) -> Optional[AudioFeatures]:
        if len(audio_block) < self.sample_rate:
            return None
            
        return self.feature_extractor.extract_features(audio_block)

    
    def load_csv_training_data(self, path: str) -> tuple[np.ndarray, np.ndarray]:
        data = np.loadtxt(path, delimiter=',', dtype=audiofeatures, skiprows=1)

        X = np.stack((
            data['chroma_stft'],
            data['rms'],
            data['spectral_centroid'],
            data['spectral_bandwidth'],
            data['rolloff'],
            data['zero_crossing_rate'],
            data['mfcc1'],
            data['mfcc2'],
            data['mfcc3'],
            data['mfcc4'],
            data['mfcc5'],
            data['mfcc6'],
            data['mfcc7'],
            data['mfcc8'],
            data['mfcc9'],
            data['mfcc10'],
            data['mfcc11'],
            data['mfcc12'],
            data['mfcc13'],
            data['mfcc14'],
            data['mfcc15'],
            data['mfcc16'],
            data['mfcc17'],
            data['mfcc18'],
            data['mfcc19'],
            data['mfcc20'],
        ), axis=1).astype(float)

        y = np.array([row[-1].decode('utf-8') for row in data])
        y = np.where(y == "FAKE", 0, 1).astype(int)
    
        return (X, y)