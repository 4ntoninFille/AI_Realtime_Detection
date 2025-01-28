import numpy as np

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