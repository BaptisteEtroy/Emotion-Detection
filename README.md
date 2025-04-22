# Emotion Detection in Text

A hybrid transformer-based model for multi-label emotion detection in text. This project combines transformer-based language models with psycholinguistic features to achieve more nuanced emotion recognition. Made for efficient emotion detection in modern conversations oover social media and online conversations.

## Features

- **28 Emotion Categories**: Detects a wide range of emotions (see the emotion mapping json)
- **Multi-label Classification**: Can identify multiple emotions in a single text
- **Hybrid Architecture**: Combines RoBERTa transformer with BiLSTM for psycholinguistic features
- **Attention Mechanism**: Uses dual attention to focus on relevant parts of input
- **Optimized Thresholds**: Per-emotion threshold optimization for improved F1 scores
- **Focal Loss**: Advanced loss function to handle class imbalance

## Installation

1. Clone this repository:
```bash
git clone https://github.com/BaptisteEtroy/emotion-detection.git
cd emotion-detection
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Preprocessing Data

First run the preprocessing python file:

```bash
python preprocessing.py
```

This will:
1. Download the GoEmotions dataset
2. Clean and preprocess the text data
3. Extract psycholinguistic features
4. Save the processed data into a `processed_data/` directory

### Training the Model

To train the model run:

```bash
python model.py
```

This will:
1. Load the preprocessed data from `processed_data/`
2. Train the hybrid emotion detection model
3. Save the trained model to `models/improved_hybrid_emotion_model.pt`
4. Generate evaluation metrics in `results/`

### Predicting Emotions

To predict emotions in new text:

```python
from model import predict_emotions

text = "I'm so excited about this new project, although I'm a bit nervous too."
emotions = predict_emotions(text)
print(emotions)
```

## Model Architecture

The model uses a hybrid architecture:

```
Input Text → RoBERTa Transformer → Attention → 
                                              → Fusion Layer → Classifier → Emotions
Psycholinguistic Features → BiLSTM → Attention →
```

- **Transformer**: Pre-trained RoBERTa model fine-tuned for emotion detection
- **BiLSTM**: Processes psycholinguistic features extracted from text
- **Attention Layers**: Help focus on relevant parts of both inputs
- **Fusion Layer**: Combines transformer and BiLSTM outputs
- **Classifier**: Multi-label classification head with sigmoid activation

## Performance

The model achieves:
- Micro F1-score: ~0.70
- Macro F1-score: ~0.60

Performance varies by emotion, with generally better results for common emotions.

## Dataset

This project uses the [GoEmotions](https://github.com/google-research/google-research/tree/master/goemotions) dataset, which contains 58k Reddit comments labeled with 28 emotions.

## License

MIT License