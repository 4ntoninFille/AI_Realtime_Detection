import logging
import kagglehub
import os


from src.models.model_factory import ModelFactory
from src.preprocessing.audio_processor import AudioProcessor
from src.utils.display import display_predictions_timeline

logger = logging.getLogger(__name__)

def main():
    logging.basicConfig(level=logging.INFO)
    logger.info('Starting AI speech recognition')

    data_path = kagglehub.dataset_download("birdy654/deep-voice-deepfake-voice-recognition")
    pathToCsv = os.path.join(data_path, "KAGGLE/DATASET-balanced.csv")
    print(data_path)

    # rf_model = ModelFactory.get_model('random_forest')
    xgb_model = ModelFactory.get_model('xgboost')

    audio_processor = AudioProcessor();
    X, y = audio_processor.load_csv_training_data(pathToCsv);

    xgb_model.train(X, y)
    # rf_model.train(X, y)

    test_sample = audio_processor.load_custome_audio("test_sample/True/musk-original.wav");
    print(test_sample)

    predictions, probabilities = xgb_model.predict(test_sample)

    display_predictions_timeline(predictions, probabilities)

if __name__ == '__main__':
    main()