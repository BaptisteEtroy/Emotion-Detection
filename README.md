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

To predict emotions in new text, modify testing.py with your desired conversation then run:

```bash
python testing.py
```
On top of the output given you will get a progression analysis saved in testing/ with different emotions and their spikes throughout the conversation.

## Repository Structure

```
Emotion-Detection/
├── models/                # will be created when running model.py (holds the models)
├── processed_data/        # will be created when running preprocessing.py(holds preprocessed datasets)
|
├── results/               # contains simple evaluation results and metrics training
├── testing/               # Output from testing.py: holds emotion progression over a conversation
├── evaluation_reports/    # Contains detailed evaluation reports with visualisations and metrics
├── visualisations/        # includes visualisation of emotion analysis over different phrases
|
├── emotion_mapping.json   # Detailed mapping of emotions with their descriptions
|
├── preprocessing.py       # Script preprocessing and feature extraction
├── model.py               # Hybrid emotion detection model training
├── evaluation.py          # Script for model evaluation
├── testing.py             # Script for analysing emotion progression over time
|
├── requirements.txt
└── README.md
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