# AI_Realtime_Detection

## Introduction

This repository contains an implementation inspired by the paper:

> [REAL-TIME DETECTION OF AI-GENERATED SPEECH FOR DEEPFAKE VOICE CONVERSION](https://arxiv.org/pdf/2308.12734v1)

The goal of this project is to develop a real-time system capable of detecting AI-generated speech, helping to counter deepfake voice conversion threats.

## Features
- **Real-time detection** of AI-generated speech.
- **Lightweight and efficient** implementation.
- **Custom Audio analyses** implementation.

## Installation

To set up the environment and install dependencies, follow these steps:

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AI_Realtime_Detection.git
   cd AI_Realtime_Detection
   ```

2. Create a virtual environment:
   ```sh
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On macOS/Linux:
     ```sh
     source venv/bin/activate
     ```
   - On Windows:
     ```sh
     venv\Scripts\activate
     ```

4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage

To run the real-time AI-generated speech detection model:

```sh
python main.py
```

## Dataset

The model requires a dataset containing real and AI-generated speech samples. I used the original dataset used in the paper [here](https://www.kaggle.com/datasets/birdy654/deep-voice-deepfake-voice-recognition), it's downloaded at the launch of the program.

## Models

There are two main models implemented right now, XGBoost and RandomForest. The code is modular, so you can implement other models by using the `ModelBase` interface.

## Results

The following results were obtained after performing 10-fold cross-validation on the training dataset:

### XGBoost Metrics:

| Class  | Precision | Recall  | F1-Score | MCC    | ROC    |
|--------|-----------|--------|----------|--------|--------|
| Real   | 0.99000  | 0.99497 | 0.99248  | 0.98471 | 0.99231 |
| Fake   | 0.99480  | 0.98965 | 0.99222  | 0.98471 | 0.99231 |
| **Weighted Average** | 0.99236  | 0.99235 | 0.99235  | 0.98471 | 0.99231 |

- **Average inference time per block:** 0.000s  
- **Maximum inference time:** 0.00078s  
- **Minimum inference time:** 0.00029s  

### RandomForest Metrics:

| Class  | Precision | Recall  | F1-Score | MCC    | ROC    |
|--------|-----------|--------|----------|--------|--------|
| Real   | 0.99494  | 0.98995 | 0.99244  | 0.98471 | 0.99238 |
| Fake   | 0.98970  | 0.99482 | 0.99226  | 0.98471 | 0.99238 |
| **Weighted Average** | 0.99236  | 0.99235 | 0.99235  | 0.98471 | 0.99238 |

- **Average inference time per block:** 0.010s  
- **Maximum inference time:** 0.01163s  
- **Minimum inference time:** 0.00952s  

These results are similar to those reported in the paper. However, it's important to note that these metrics are based on the training dataset. The model's performance may be less precise when applied to other, unseen audio samples.

## Contributions

Contributions are welcome! Feel free to open issues or submit pull requests to improve the project.

