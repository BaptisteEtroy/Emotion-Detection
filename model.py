import os
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, precision_recall_curve
import logging
import json
from tqdm import tqdm
import argparse

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set device
device = torch.device("mps")
logger.info(f"Using device: {device}")

# Constants - Updated for better performance
TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-emotion"  # Emotion-specific pre-trained model
MAX_LEN = 128
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch size of 64
EPOCHS = 20  # Increased for better convergence
LEARNING_RATE = 1e-5
NUM_WARMUP_STEPS = 200  # More warmup steps
PSYCHOLINGUISTIC_FEATURES_DIM = 20
LSTM_HIDDEN_DIM = 128
DROPOUT_RATE = 0.3
EARLY_STOPPING_PATIENCE = 3  # patience for early stopping
FOCAL_LOSS_GAMMA = 2.0  # For focal loss
WEIGHT_DECAY = 0.01  # L2 regularization
OPTIMIZER = "AdamW"  # Can be "AdamW" or "Adam"
SCHEDULER = "linear_warmup"  # Can be "linear_warmup" or "cosine_warmup"
USE_LABEL_SMOOTHING = True  # Use label smoothing for better generalization
LABEL_SMOOTHING = 0.1  # Label smoothing factor

# Use three different emotion-specialized transformers in parallel
base_models = [
    "cardiffnlp/twitter-roberta-base-emotion",
    "bhadresh-savani/distilbert-base-uncased-emotion",
    "arpanghoshal/EmoRoBERTa"
]

class EmotionDataset(Dataset):
    def __init__(self, texts, labels=None, psycholinguistic_features=None, tokenizer=None):
        self.texts = texts
        self.labels = labels
        self.psycholinguistic_features = psycholinguistic_features
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }
        
        if self.psycholinguistic_features is not None:
            item['psycholinguistic_features'] = torch.tensor(
                self.psycholinguistic_features[idx], dtype=torch.float
            )
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
            
        return item

class AttentionLayer(nn.Module):
    """Simple attention mechanism for focusing on important features"""
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attention(x)  # (batch_size, seq_len, 1)
        context = torch.sum(attn_weights * x, dim=1)  # (batch_size, hidden_dim)
        return context, attn_weights

class HybridEmotionModel(nn.Module):
    def __init__(self, num_labels, psycholinguistic_dim=PSYCHOLINGUISTIC_FEATURES_DIM):
        super(HybridEmotionModel, self).__init__()
        
        # Transformer Layer - using emotion-specific pre-trained model
        self.transformer = AutoModel.from_pretrained(TRANSFORMER_MODEL)
        self.transformer_dim = self.transformer.config.hidden_size
        
        # BiLSTM Layer for psycholinguistic features
        self.bilstm = nn.LSTM(
            input_size=psycholinguistic_dim,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=DROPOUT_RATE if DROPOUT_RATE > 0 else 0
        )
        
        # Attention mechanisms for both transformer and BiLSTM outputs
        self.transformer_attention = AttentionLayer(self.transformer_dim)
        self.bilstm_attention = AttentionLayer(2*LSTM_HIDDEN_DIM)
        
        # Fusion layer with higher capacity
        self.fusion = nn.Sequential(
            nn.Linear(self.transformer_dim + 2*LSTM_HIDDEN_DIM, self.transformer_dim),
            nn.LayerNorm(self.transformer_dim),
            nn.Dropout(DROPOUT_RATE),
            nn.ReLU(),
            nn.Linear(self.transformer_dim, self.transformer_dim // 2),
            nn.Dropout(DROPOUT_RATE),
            nn.ReLU()
        )
        
        # Dropout for regularization
        self.dropout = nn.Dropout(DROPOUT_RATE)
        
        # Multi-label classification head
        self.classifier = nn.Linear(self.transformer_dim // 2, num_labels)
        
        # Sigmoid activation for multi-label output
        self.sigmoid = nn.Sigmoid()
        
        # Initialize weights properly
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights for better training dynamics"""
        # Initialize fusion layers
        for module in self.fusion.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        
        # Initialize classifier
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.constant_(self.classifier.bias, 0)
        
        # Initialize attention layers
        for module in [self.transformer_attention, self.bilstm_attention]:
            for submodule in module.modules():
                if isinstance(submodule, nn.Linear):
                    nn.init.xavier_uniform_(submodule.weight)
                    if submodule.bias is not None:
                        nn.init.constant_(submodule.bias, 0)
        
    def forward(self, input_ids, attention_mask, psycholinguistic_features=None, apply_sigmoid=True):
        # Process text through transformer
        transformer_outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get hidden states from transformer (last layer)
        hidden_states = transformer_outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # Apply attention to get context-aware representation
        transformer_context, _ = self.transformer_attention(hidden_states)
        
        if psycholinguistic_features is not None:
            # Process psycholinguistic features through BiLSTM
            psycho_features = psycholinguistic_features.unsqueeze(1)
            lstm_outputs, _ = self.bilstm(psycho_features)
            
            # Apply attention to BiLSTM outputs
            lstm_context, _ = self.bilstm_attention(lstm_outputs)
            
            # Concatenate transformer context with BiLSTM context
            combined = torch.cat((transformer_context, lstm_context), dim=1)
            
            # Pass through fusion layer
            fused_output = self.fusion(combined)
        else:
            # If no psycholinguistic features, use only transformer output
            fused_output = self.fusion(
                torch.cat((transformer_context, torch.zeros(transformer_context.size(0), 2*LSTM_HIDDEN_DIM).to(device)), dim=1)
            )
        
        # Apply dropout for regularization
        fused_output = self.dropout(fused_output)
        
        # Pass through classifier
        logits = self.classifier(fused_output)
        
        # Apply sigmoid for multi-label classification if requested
        if apply_sigmoid:
            return self.sigmoid(logits)
        else:
            return logits

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced multi-label classification"""
    def __init__(self, gamma=FOCAL_LOSS_GAMMA, alpha=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha  # weight for each class
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        BCE_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=self.alpha)(inputs, targets)
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        F_loss = (1-pt)**self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

def smooth_labels(labels, smoothing=0.1):
    """
    Apply label smoothing to the one-hot encoded labels.
    Args:
        labels: One-hot encoded labels tensor
        smoothing: Smoothing factor (default: 0.1)
    Returns:
        Smoothed labels tensor
    """
    assert 0 <= smoothing < 1
    return labels * (1 - smoothing) + smoothing / labels.size(1)

def calculate_class_weights(labels):
    """Calculate class weights inversely proportional to class frequency"""
    # Count positive samples for each class
    pos_counts = np.sum(labels, axis=0)
    
    # Calculate weights (inversely proportional to frequency)
    weights = len(labels) / (len(np.unique(pos_counts)) * (pos_counts + 1e-5))
    
    # Normalize weights to have mean = 1
    weights = weights * len(weights) / weights.sum()
    
    # Cap weights to avoid extremely large values (max 10.0)
    weights = np.minimum(weights, 10.0)
    
    logger.info(f"Class weights range: {weights.min():.4f} - {weights.max():.4f}")
    
    return torch.tensor(weights, dtype=torch.float32)

def optimize_thresholds(model, val_dataloader, device):
    """Optimize classification thresholds for each class using validation set"""
    logger.info("Optimizing classification thresholds...")
    model.eval()
    all_outputs = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Threshold optimization"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            psycho_features = batch.get('psycholinguistic_features', None)
            
            if psycho_features is not None:
                psycho_features = psycho_features.to(device)
            
            # Raw predictions without sigmoid
            raw_outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                psycholinguistic_features=psycho_features,
                apply_sigmoid=False  # Get raw logits
            )
            
            all_outputs.append(raw_outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate all outputs and labels
    all_outputs = np.vstack(all_outputs)
    all_labels = np.vstack(all_labels)
    
    # Find optimal threshold for each class
    num_classes = all_outputs.shape[1]
    thresholds = np.zeros(num_classes)
    
    for i in range(num_classes):
        # Skip if no positive samples for this class
        if np.sum(all_labels[:, i]) == 0:
            thresholds[i] = 0.5  # Default
            continue
            
        # Get precision-recall curve
        precision, recall, threshold = precision_recall_curve(all_labels[:, i], all_outputs[:, i])
        
        # Calculate F1 score for each threshold
        f1_scores = 2 * recall * precision / (recall + precision + 1e-8)
        
        # Find threshold with highest F1 score
        if len(threshold) > 0:
            best_idx = np.argmax(f1_scores[:-1])  # Last element has no threshold
            thresholds[i] = threshold[best_idx]
        else:
            thresholds[i] = 0.5  # Default
    
    logger.info(f"Optimized thresholds range: {thresholds.min():.4f} - {thresholds.max():.4f}")
    return thresholds

def extract_psycholinguistic_features(texts):
    """Extract basic psycholinguistic features from text"""
    # Initialize features array
    features = np.zeros((len(texts), PSYCHOLINGUISTIC_FEATURES_DIM))
    
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            text = str(text)
        
        # Basic text metrics
        words = text.split()
        chars = list(text)
        
        # Feature 1: Text length (character count)
        features[i, 0] = len(chars)
        
        # Feature 2: Word count
        features[i, 1] = len(words)
        
        # Feature 3: Average word length
        if len(words) > 0:
            features[i, 2] = sum(len(w) for w in words) / len(words)
        
        # Feature 4: Sentence count (rough approximation)
        sentences = text.split('.')
        features[i, 3] = len([s for s in sentences if len(s.strip()) > 0])
        
        # Feature 5: Average sentence length in words
        if features[i, 3] > 0:
            features[i, 4] = features[i, 1] / features[i, 3]
        
        # Feature 6: Punctuation frequency
        features[i, 5] = sum(1 for c in chars if c in '.,;:!?()[]{}"\'')
        
        # Feature 7: Uppercase letter frequency
        features[i, 6] = sum(1 for c in chars if c.isupper())
        
        # Feature 8: Digit frequency
        features[i, 7] = sum(1 for c in chars if c.isdigit())
        
        # Feature 9: Special character frequency
        features[i, 8] = sum(1 for c in chars if not c.isalnum() and not c.isspace())
        
        # Feature 10: Ratio of uppercase to lowercase
        lowercase_count = sum(1 for c in chars if c.islower())
        if lowercase_count > 0:
            features[i, 9] = features[i, 6] / lowercase_count
        
        # Feature 11-15: Word length distributions
        word_lengths = [len(w) for w in words]
        if len(words) > 0:
            features[i, 10] = sum(1 for l in word_lengths if 1 <= l <= 5) / len(words)
            features[i, 11] = sum(1 for l in word_lengths if 6 <= l <= 10) / len(words)
            features[i, 12] = sum(1 for l in word_lengths if 11 <= l <= 15) / len(words)
            features[i, 13] = sum(1 for l in word_lengths if 16 <= l <= 20) / len(words)
            features[i, 14] = sum(1 for l in word_lengths if l > 20) / len(words)
        
        # Fill remaining features with zeros
        features[i, 15:] = np.zeros(PSYCHOLINGUISTIC_FEATURES_DIM - 15)
    
    # Normalize features
    epsilon = 1e-8
    feature_means = np.mean(features, axis=0)
    feature_stds = np.std(features, axis=0) + epsilon
    normalized_features = (features - feature_means) / feature_stds
    
    return normalized_features

def parse_emotion_ids(emotion_id_str):
    """Parse comma-separated emotion IDs into a list of integers"""
    if pd.isna(emotion_id_str):
        return []
    
    # Convert to string to handle both string and numeric types
    emotion_id_str = str(emotion_id_str)
    
    # Handle both single integers and comma-separated lists
    if ',' in emotion_id_str:
        # Split and filter out any empty or non-numeric items
        return [int(eid.strip()) for eid in emotion_id_str.split(',') 
               if eid.strip() and eid.strip().replace('.', '', 1).isdigit()]
    elif emotion_id_str.strip().replace('.', '', 1).isdigit():
        # Single numeric value
        return [int(float(emotion_id_str))]
    else:
        logger.warning(f"Non-numeric emotion_id: {emotion_id_str}")
        return []

def prepare_data(train_df, val_df, test_df, tokenizer):
    """Prepare data for the model - convert to multi-label format and create datasets"""
    # Extract text data
    train_texts = train_df['preprocessed_text'].values
    val_texts = val_df['preprocessed_text'].values
    test_texts = test_df['preprocessed_text'].values
    
    # Extract emotion IDs and convert to one-hot encoding
    train_emotion_lists = [parse_emotion_ids(eid) for eid in train_df['emotion_id'].values]
    val_emotion_lists = [parse_emotion_ids(eid) for eid in val_df['emotion_id'].values]
    test_emotion_lists = [parse_emotion_ids(eid) for eid in test_df['emotion_id'].values]
    
    # Find the total number of unique emotions
    all_emotions = set()
    for emotion_list in train_emotion_lists + val_emotion_lists + test_emotion_lists:
        all_emotions.update(emotion_list)
    
    num_classes = max(all_emotions) + 1 if all_emotions else 1
    logger.info(f"Detected {num_classes} emotion classes")
    
    # Convert to one-hot encoding
    train_labels_onehot = np.zeros((len(train_emotion_lists), num_classes))
    val_labels_onehot = np.zeros((len(val_emotion_lists), num_classes))
    test_labels_onehot = np.zeros((len(test_emotion_lists), num_classes))
    
    # Fill in the one-hot vectors
    for i, emotions in enumerate(train_emotion_lists):
        for emotion in emotions:
            if 0 <= emotion < num_classes:
                train_labels_onehot[i, emotion] = 1
            
    for i, emotions in enumerate(val_emotion_lists):
        for emotion in emotions:
            if 0 <= emotion < num_classes:
                val_labels_onehot[i, emotion] = 1
            
    for i, emotions in enumerate(test_emotion_lists):
        for emotion in emotions:
            if 0 <= emotion < num_classes:
                test_labels_onehot[i, emotion] = 1
    
    # Load emotion mapping from file
    if os.path.exists('emotion_mapping.json'):
        with open('emotion_mapping.json', 'r') as f:
            emotion_mapping = json.load(f)
        logger.info(f"Loaded emotion mapping from {emotion_mapping}")
        
        # Create emotion class names using the mapping
        emotion_classes = []
        for i in range(num_classes):
            str_i = str(i)
            if str_i in emotion_mapping:
                emotion_classes.append(emotion_mapping[str_i])
            else:
                emotion_classes.append(f"emotion_{i}")
    else:
        emotion_classes = [f"emotion_{i}" for i in range(num_classes)]
    
    # Save emotion classes for later use
    with open('emotion_classes.json', 'w') as f:
        json.dump(list(emotion_classes), f)
    
    # Also save with the correct mapping
    print("Saving emotion mapping...")
    emotion_mapping = {str(i): emotion_name for i, emotion_name in enumerate(emotion_classes)}
    with open('emotion_mapping.json', 'w') as f:
        json.dump(emotion_mapping, f, indent=4)
    print(f"✓ Saved emotion mapping with {len(emotion_mapping)} emotions")
    
    logger.info(f"Prepared labels: train={train_labels_onehot.shape}, val={val_labels_onehot.shape}, test={test_labels_onehot.shape}")
    
    # Extract psycholinguistic features
    train_psycho = extract_psycholinguistic_features(train_texts)
    val_psycho = extract_psycholinguistic_features(val_texts)
    test_psycho = extract_psycholinguistic_features(test_texts)
    
    # Create datasets
    train_dataset = EmotionDataset(
        texts=train_texts,
        labels=train_labels_onehot,
        psycholinguistic_features=train_psycho,
        tokenizer=tokenizer
    )
    
    val_dataset = EmotionDataset(
        texts=val_texts,
        labels=val_labels_onehot,
        psycholinguistic_features=val_psycho,
        tokenizer=tokenizer
    )
    
    test_dataset = EmotionDataset(
        texts=test_texts,
        labels=test_labels_onehot,
        psycholinguistic_features=test_psycho,
        tokenizer=tokenizer
    )
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE
    )
    
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE
    )
    
    return train_dataloader, val_dataloader, test_dataloader, len(emotion_classes)

def augment_data(texts, labels):
    # Add back-translation, synonym replacement, word deletion
    augmented_texts = []
    augmented_labels = []
    # Implementation here
    return augmented_texts, augmented_labels

def train_model(model, train_dataloader, val_dataloader, optimizer, scheduler, device, class_weights=None, epochs=EPOCHS):
    """Train the model with early stopping and threshold optimization"""
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    thresholds = np.ones(model.classifier.out_features) * 0.5  # Default thresholds
    
    # Define loss function for multi-label classification
    if class_weights is not None:
        class_weights = class_weights.to(device)
        if USE_LABEL_SMOOTHING:
            criterion = FocalLoss(alpha=class_weights, label_smoothing=LABEL_SMOOTHING)
            logger.info(f"Using Focal Loss with class weights and label smoothing {LABEL_SMOOTHING}")
        else:
            criterion = FocalLoss(alpha=class_weights)
            logger.info("Using Focal Loss with class weights")
    else:
        if USE_LABEL_SMOOTHING:
            criterion = nn.BCEWithLogitsLoss(label_smoothing=LABEL_SMOOTHING)
            logger.info(f"Using BCEWithLogitsLoss with label smoothing {LABEL_SMOOTHING}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            logger.info("Using BCEWithLogitsLoss")
    
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch+1}/{epochs}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_dataloader, desc="Training")
        optimizer.zero_grad()  # Reset gradients at the beginning of epoch
        batch_count = 0
        
        for batch in progress_bar:
            batch_count += 1
            
            # Move batch to device
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            psycho_features = batch.get('psycholinguistic_features', None)
            
            if psycho_features is not None:
                psycho_features = psycho_features.to(device)
            
            # Forward pass - get raw logits for loss function
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                psycholinguistic_features=psycho_features,
                apply_sigmoid=False  # Get raw logits for BCE loss
            )
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Normalize loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS
            
            # Backward pass
            loss.backward()
            
            # Update weights only after accumulating gradients for several batches
            if batch_count % GRADIENT_ACCUMULATION_STEPS == 0 or batch_count == len(train_dataloader):
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                # Update parameters
                optimizer.step()
                
                # Update scheduler
                scheduler.step()
                
                # Reset gradients
                optimizer.zero_grad()
            
            # Update running loss (use the non-normalized loss for reporting)
            train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_dataloader)
        logger.info(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                psycho_features = batch.get('psycholinguistic_features', None)
                
                if psycho_features is not None:
                    psycho_features = psycho_features.to(device)
                
                # Forward pass - get raw logits for loss and classification
                raw_outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    psycholinguistic_features=psycho_features,
                    apply_sigmoid=False
                )
                
                # Calculate loss with raw logits
                loss = criterion(raw_outputs, labels)
                
                # Update running loss
                val_loss += loss.item()
                
                # Apply sigmoid to get probabilities
                probs = torch.sigmoid(raw_outputs)
                
                # Store raw outputs for threshold optimization
                all_outputs.append(raw_outputs.cpu().numpy())
                
                # Apply current thresholds for predictions
                preds = torch.zeros_like(probs)
                for i in range(probs.size(1)):
                    preds[:, i] = (probs[:, i] > thresholds[i]).float()
                
                # Append to lists for metrics calculation
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(val_dataloader)
        
        # Concatenate predictions and labels
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        
        # Calculate F1 scores
        val_f1_micro = f1_score(all_labels, all_preds, average='micro')
        val_f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        logger.info(f"Validation Loss: {avg_val_loss:.4f}, Micro-F1: {val_f1_micro:.4f}, Macro-F1: {val_f1_macro:.4f}")
        
        # Optimize thresholds every second epoch
        if epoch % 2 == 1 or epoch == epochs - 1:
            # Convert raw outputs to probabilities for threshold optimization
            all_probs = 1 / (1 + np.exp(-all_outputs))
            new_thresholds = np.zeros(all_probs.shape[1])
            
            # Find optimal threshold for each class
            for i in range(all_probs.shape[1]):
                if np.sum(all_labels[:, i]) > 0:  # Only optimize if there are positive samples
                    precision, recall, thresh = precision_recall_curve(all_labels[:, i], all_probs[:, i])
                    f1 = 2 * recall * precision / (recall + precision + 1e-8)
                    if len(thresh) > 0:
                        best_idx = np.argmax(f1[:-1])  # Last element has no threshold
                        new_thresholds[i] = thresh[best_idx]
                    else:
                        new_thresholds[i] = 0.5
                else:
                    new_thresholds[i] = 0.5
            
            # Update thresholds
            thresholds = new_thresholds
            logger.info(f"Updated thresholds - min: {thresholds.min():.3f}, max: {thresholds.max():.3f}, avg: {thresholds.mean():.3f}")
            
            # Recalculate predictions with new thresholds
            all_preds = np.zeros_like(all_probs)
            for i in range(all_probs.shape[1]):
                all_preds[:, i] = (all_probs[:, i] > thresholds[i]).astype(float)
            
            # Recalculate F1 scores
            new_val_f1_micro = f1_score(all_labels, all_preds, average='micro')
            new_val_f1_macro = f1_score(all_labels, all_preds, average='macro')
            
            logger.info(f"With optimized thresholds - Micro-F1: {new_val_f1_micro:.4f}, Macro-F1: {new_val_f1_macro:.4f}")
            
            # Use the optimized F1 score
            val_f1 = new_val_f1_micro
        else:
            val_f1 = val_f1_micro
        
        # Save best model
        if val_f1 > best_val_f1:
            logger.info(f"Validation F1 improved from {best_val_f1:.4f} to {val_f1:.4f}")
            best_val_f1 = val_f1
            best_model_state = {
                'model_state_dict': model.state_dict(),
                'thresholds': thresholds,
                'class_weights': class_weights.cpu().numpy() if class_weights is not None else None,
                'val_f1_micro': val_f1_micro,
                'val_f1_macro': val_f1_macro
            }
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
        
        # Early stopping
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping after {epoch+1} epochs")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state['model_state_dict'])
        logger.info(f"Loaded best model with validation F1: {best_val_f1:.4f}")
    
    return model, best_model_state

def cross_validation(train_df, val_df, tokenizer, n_splits=5, batch_size=16):
    """
    Perform cross-validation evaluation
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        tokenizer: Tokenizer for text processing
        n_splits: Number of CV folds
        batch_size: Batch size for evaluation
        
    Returns:
        cv_results: Dictionary with cross-validation results
    """
    logger.info(f"Performing {n_splits}-fold cross-validation...")
    
    # Combine train and validation sets for CV
    combined_df = pd.concat([train_df, val_df], ignore_index=True)
    
    # Extract texts and emotion IDs
    texts = combined_df['preprocessed_text'].values
    emotion_lists = [parse_emotion_ids(eid) for eid in combined_df['emotion_id'].values]
    
    # Find the total number of unique emotions
    all_emotions = set()
    for emotion_list in emotion_lists:
        all_emotions.update(emotion_list)
    
    num_classes = max(all_emotions) + 1 if all_emotions else 1
    logger.info(f"Detected {num_classes} emotion classes")
    
    # Convert to one-hot encoding
    labels_onehot = np.zeros((len(emotion_lists), num_classes))
    for i, emotions in enumerate(emotion_lists):
        for emotion in emotions:
            if 0 <= emotion < num_classes:
                labels_onehot[i, emotion] = 1
    
    # Extract psycholinguistic features
    psycho_features = extract_psycholinguistic_features(texts)
    
    # Setup cross-validation
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    cv_metrics = []
    
    # Perform cross-validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(texts)))):
        logger.info(f"\nFold {fold+1}/{n_splits}")
        
        # Split data for this fold
        X_train, X_val = texts[train_idx], texts[val_idx]
        y_train, y_val = labels_onehot[train_idx], labels_onehot[val_idx]
        p_train, p_val = psycho_features[train_idx], psycho_features[val_idx]
        
        # Create datasets
        train_dataset = EmotionDataset(
            texts=X_train,
            labels=y_train,
            psycholinguistic_features=p_train,
            tokenizer=tokenizer
        )
        
        val_dataset = EmotionDataset(
            texts=X_val,
            labels=y_val,
            psycholinguistic_features=p_val,
            tokenizer=tokenizer
        )
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Create and train model
        model = HybridEmotionModel(num_labels=num_classes)
        model.to(device)
        
        # Initialize optimizer and scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=100)
        
        # Train for 3 epochs (faster for CV)
        for epoch in range(3):
            # Training
            model.train()
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/3 (Training)"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                psycho_features = batch.get('psycholinguistic_features', None)
                
                if psycho_features is not None:
                    psycho_features = psycho_features.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    psycholinguistic_features=psycho_features,
                    apply_sigmoid=False
                )
                
                # Calculate loss
                criterion = nn.BCEWithLogitsLoss()
                loss = criterion(outputs, labels)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
        
        # Evaluate on validation set
        # Initialize arrays for predictions and ground truth
        all_probs = []
        all_preds = []
        all_labels = []
        
        # Evaluate model
        model.eval()
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Evaluating"):
                # Move batch to device
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                psycho_features = batch.get('psycholinguistic_features', None)
                
                if psycho_features is not None:
                    psycho_features = psycho_features.to(device)
                
                # Forward pass
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    psycholinguistic_features=psycho_features,
                    apply_sigmoid=True
                )
                
                # Use default thresholds of 0.5
                preds = torch.zeros_like(outputs)
                for i in range(outputs.size(1)):
                    preds[:, i] = (outputs[:, i] > 0.5).float()
                
                # Store predictions and labels
                all_probs.append(outputs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        
        # Concatenate results
        all_probs = np.vstack(all_probs)
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
        # Optimize thresholds
        thresholds = np.zeros(num_classes)
        for i in range(num_classes):
            # Skip if no positive samples
            if np.sum(all_labels[:, i]) == 0:
                thresholds[i] = 0.5
                continue
            
            # Calculate precision-recall curve
            precision, recall, thresh = precision_recall_curve(all_labels[:, i], all_probs[:, i])
            
            # Calculate F1 score for each threshold
            f1 = 2 * precision * recall / (precision + recall + 1e-8)
            
            # Find threshold with highest F1 score
            if len(thresh) > 0:
                best_idx = np.argmax(f1[:-1])  # Last element has no threshold
                thresholds[i] = thresh[best_idx]
            else:
                thresholds[i] = 0.5
        
        # Update predictions with optimized thresholds
        for i in range(all_probs.shape[1]):
            all_preds[:, i] = (all_probs[:, i] > thresholds[i]).astype(float)
        
        # Calculate metrics
        from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
        
        fold_metrics = {}
        fold_metrics['fold'] = fold + 1
        
        # Overall metrics
        fold_metrics['micro_f1'] = f1_score(all_labels, all_preds, average='micro')
        fold_metrics['macro_f1'] = f1_score(all_labels, all_preds, average='macro')
        fold_metrics['weighted_f1'] = f1_score(all_labels, all_preds, average='weighted')
        fold_metrics['samples_f1'] = f1_score(all_labels, all_preds, average='samples')
        
        fold_metrics['micro_precision'] = precision_score(all_labels, all_preds, average='micro')
        fold_metrics['macro_precision'] = precision_score(all_labels, all_preds, average='macro')
        
        fold_metrics['micro_recall'] = recall_score(all_labels, all_preds, average='micro')
        fold_metrics['macro_recall'] = recall_score(all_labels, all_preds, average='macro')
        
        # Sample-based metrics
        fold_metrics['hamming_loss'] = hamming_loss(all_labels, all_preds)
        fold_metrics['exact_match_ratio'] = np.mean(np.all(all_labels == all_preds, axis=1))
        
        logger.info(f"Fold {fold+1} results: Micro-F1 = {fold_metrics['micro_f1']:.4f}, Macro-F1 = {fold_metrics['macro_f1']:.4f}")
        
        cv_metrics.append(fold_metrics)
    
    # Calculate average metrics across folds
    avg_metrics = {}
    for key in cv_metrics[0].keys():
        if key == 'fold':
            continue
        avg_metrics[key] = np.mean([m[key] for m in cv_metrics])
    
    # Log average metrics
    logger.info("\nCross-validation average results:")
    logger.info(f"Micro-F1 = {avg_metrics['micro_f1']:.4f}, Macro-F1 = {avg_metrics['macro_f1']:.4f}")
    logger.info(f"Hamming Loss = {avg_metrics['hamming_loss']:.4f}, Exact Match = {avg_metrics['exact_match_ratio']:.4f}")
    
    return {
        'fold_metrics': cv_metrics,
        'average_metrics': avg_metrics
    }

def main():
    """Main function to train and save the model"""
    start_time = time.time()
    logger.info("Starting emotion detection model training...")
    
    # Check for model directory
    os.makedirs("models", exist_ok=True)
    
    # Load the tokenizer
    logger.info(f"Loading tokenizer from {TRANSFORMER_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_MODEL)
    
    # Load data
    logger.info("Loading data...")
    train_df = pd.read_csv("processed_data/train.csv")
    val_df = pd.read_csv("processed_data/val.csv")
    test_df = pd.read_csv("processed_data/test.csv")
    
    # Prepare data
    logger.info("Preparing data...")
    train_dataloader, val_dataloader, test_dataloader, num_labels = prepare_data(
        train_df, val_df, test_df, tokenizer
    )
    
    # Parse command line arguments to determine whether to train or run cross-validation
    parser = argparse.ArgumentParser(description='Emotion Detection Model')
    parser.add_argument('--cv', action='store_true', help='Run cross-validation')
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of training epochs')
    args = parser.parse_args()
    
    if args.cv:
        # Run cross-validation
        logger.info("Running cross-validation...")
        cv_results = cross_validation(train_df, val_df, tokenizer)
        
        # Save cross-validation results
        cv_results_path = "models/cross_validation_results.json"
        with open(cv_results_path, "w") as f:
            json.dump(cv_results, f, indent=4)
        
        logger.info(f"Cross-validation results saved to {cv_results_path}")
        
    else:
        # Create model
        logger.info(f"Creating model with {num_labels} emotion classes...")
        model = HybridEmotionModel(num_labels=num_labels)
        model.to(device)
        
        # Calculate class weights for loss function
        train_labels = []
        for batch in train_dataloader:
            train_labels.append(batch['labels'].cpu().numpy())
        train_labels = np.vstack(train_labels)
        class_weights = calculate_class_weights(train_labels)
        
        # Optimizer
        optimizer = getattr(optim, OPTIMIZER)(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY
        )
        
        # Scheduler
        num_training_steps = len(train_dataloader) * args.epochs
        
        if SCHEDULER == "linear_warmup":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=num_training_steps
            )
        elif SCHEDULER == "cosine_warmup":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=NUM_WARMUP_STEPS,
                num_training_steps=num_training_steps
            )
        
        # Train model
        logger.info(f"Training model for {args.epochs} epochs...")
        model, best_model_state = train_model(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            class_weights=class_weights,
            epochs=args.epochs
        )
        
        # Save model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save with additional metadata
        model_data = {
            'model_state_dict': model.state_dict(),
            'num_labels': num_labels,
            'transformer_model': TRANSFORMER_MODEL,
            'training_time': time.time() - start_time,
            'epochs': args.epochs,
            'batch_size': BATCH_SIZE,
            'learning_rate': LEARNING_RATE
        }
        
        model_path = "models/improved_hybrid_emotion_model.pt"
        torch.save(model_data, model_path)
        logger.info(f"Model saved to {model_path}")
    
    logger.info(f"Process completed in {time.time() - start_time:.2f} seconds")

def predict_emotions(text, model_path="models/improved_hybrid_emotion_model.pt"):
    """
    Predict emotions for a given text using the trained model
    
    Args:
        text (str): The input text to analyze
        model_path (str): Path to the saved model
        
    Returns:
        dict: Predicted emotions with their probabilities
    """
    # Load model data
    try:
        # Try with weights_only=False for compatibility with PyTorch 2.6+
        model_data = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions that don't have weights_only parameter
        model_data = torch.load(model_path, map_location=device)
    
    # Get model parameters
    num_labels = model_data['num_labels']
    transformer_name = model_data.get('transformer_model', TRANSFORMER_MODEL)
    thresholds = model_data.get('thresholds', np.ones(num_labels) * 0.5)
    
    # Load tokenizer for the correct model
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    
    # Create model
    model = HybridEmotionModel(num_labels=num_labels)
    
    # Load model weights
    model.load_state_dict(model_data['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load emotion classes
    with open('emotion_classes.json', 'r') as f:
        emotion_classes = json.load(f)
    
    # Preprocess and tokenize text
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Extract psycholinguistic features
    psycho_features = extract_psycholinguistic_features([text])
    psycho_features = torch.tensor(psycho_features, dtype=torch.float).to(device)
    
    # Move inputs to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            psycholinguistic_features=psycho_features
        )
    
    # Convert outputs to probabilities
    probs = outputs.cpu().numpy()[0]
    
    # Apply optimized thresholds for predictions
    preds = np.zeros_like(probs)
    for i in range(len(probs)):
        preds[i] = 1 if probs[i] > thresholds[i] else 0
    
    # Create dictionary of emotion probabilities and predictions
    emotion_results = {}
    for emotion, prob, pred in zip(emotion_classes, probs, preds):
        emotion_results[emotion] = {
            'probability': float(prob),
            'detected': bool(pred),
            'threshold': float(thresholds[emotion_classes.index(emotion)])
        }
    
    # Sort by probability (descending)
    emotion_results = dict(sorted(emotion_results.items(), key=lambda x: x[1]['probability'], reverse=True))
    
    return emotion_results

if __name__ == "__main__":
    main() 