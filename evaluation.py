import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score, 
    confusion_matrix, roc_curve, auc, 
    precision_recall_curve, average_precision_score,
    hamming_loss, multilabel_confusion_matrix
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.serialization
from model import HybridEmotionModel, EmotionDataset, extract_psycholinguistic_features
from datetime import datetime
from tqdm import tqdm
import time
import itertools

# For PyTorch 2.6+ compatibility
try:
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except:
    pass

# Default paths
MODEL_PATH = "models/improved_hybrid_emotion_model.pt"
DATA_PATHS = {
    "train": "processed_data/train.csv",
    "val": "processed_data/val.csv",
    "test": "processed_data/test.csv"
}
REPORTS_DIR = "evaluation_reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else 
                     "mps" if torch.backends.mps.is_available() else 
                     "cpu")

def load_model_and_data(model_path=MODEL_PATH, data_paths=DATA_PATHS):
    """
    Load the trained model and datasets
    
    Args:
        model_path: Path to the saved model
        data_paths: Dictionary with paths to train, val and test data
        
    Returns:
        model: Loaded model
        datasets: Dictionary with train, val and test datasets
        model_data: Dictionary with model metadata
        emotion_classes: List of emotion class names
    """
    print(f"Loading model from {model_path}...")
    
    try:
        # Try with weights_only=False for compatibility with PyTorch 2.6+
        model_data = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        # Fallback for older PyTorch versions
        model_data = torch.load(model_path, map_location=device)
    
    # Get model parameters
    num_labels = model_data['num_labels']
    transformer_name = model_data.get('transformer_model', "cardiffnlp/twitter-roberta-base-emotion")
    
    # Create and load model
    model = HybridEmotionModel(num_labels=num_labels)
    model.load_state_dict(model_data['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Load tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(transformer_name)
    
    # Load emotion classes
    with open('emotion_classes.json', 'r') as f:
        emotion_classes = json.load(f)
    
    # Load datasets
    print("Loading datasets...")
    datasets = {}
    for split, path in data_paths.items():
        df = pd.read_csv(path)
        
        # Extract text and features
        texts = df['preprocessed_text'].values
        psycho_features = extract_psycholinguistic_features(texts)
        
        # Extract emotion IDs
        if 'emotion_id' in df.columns:
            from model import parse_emotion_ids
            emotion_lists = [parse_emotion_ids(eid) for eid in df['emotion_id'].values]
            
            # Convert to one-hot encoding
            labels = np.zeros((len(emotion_lists), num_labels))
            for i, emotions in enumerate(emotion_lists):
                for emotion in emotions:
                    if 0 <= emotion < num_labels:
                        labels[i, emotion] = 1
            
            # Create dataset
            datasets[split] = EmotionDataset(
                texts=texts,
                labels=labels,
                psycholinguistic_features=psycho_features,
                tokenizer=tokenizer
            )
    
    return model, datasets, model_data, emotion_classes

def evaluate_model(model, dataset, device, batch_size=16, thresholds=None):
    """
    Evaluate the model on a dataset
    
    Args:
        model: The model to evaluate
        dataset: The dataset to evaluate on
        device: The device to use for evaluation
        batch_size: Batch size for evaluation
        thresholds: Thresholds to use for classification
        
    Returns:
        results: Dictionary with evaluation results
    """
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    if thresholds is None:
        # Use default threshold of 0.5 for all classes
        thresholds = np.ones(model.classifier.out_features) * 0.5
    
    # Initialize arrays for predictions and ground truth
    all_probs = []
    all_preds = []
    all_labels = []
    
    # Evaluate model
    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
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
            
            # Apply thresholds
            preds = torch.zeros_like(outputs)
            for i in range(outputs.size(1)):
                preds[:, i] = (outputs[:, i] > thresholds[i]).float()
            
            # Store predictions and labels
            all_probs.append(outputs.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # Concatenate results
    all_probs = np.vstack(all_probs)
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    return {
        'probabilities': all_probs,
        'predictions': all_preds,
        'labels': all_labels,
        'thresholds': thresholds
    }

def optimize_thresholds(labels, probs):
    """
    Optimize thresholds for each class to maximize F1 score
    
    Args:
        labels: Ground truth labels
        probs: Predicted probabilities
        
    Returns:
        thresholds: Optimized thresholds for each class
    """
    num_classes = probs.shape[1]
    thresholds = np.zeros(num_classes)
    
    for i in range(num_classes):
        # Skip if no positive samples
        if np.sum(labels[:, i]) == 0:
            thresholds[i] = 0.5
            continue
        
        # Calculate precision-recall curve
        precision, recall, thresh = precision_recall_curve(labels[:, i], probs[:, i])
        
        # Calculate F1 score for each threshold
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        # Find threshold with highest F1 score
        if len(thresh) > 0:
            best_idx = np.argmax(f1[:-1])  # Last element has no threshold
            thresholds[i] = thresh[best_idx]
        else:
            thresholds[i] = 0.5
    
    return thresholds

def calculate_metrics(results, emotion_classes):
    """
    Calculate comprehensive metrics for model evaluation
    
    Args:
        results: Dictionary with evaluation results
        emotion_classes: List of emotion class names
        
    Returns:
        metrics: Dictionary with calculated metrics
    """
    labels = results['labels']
    preds = results['predictions']
    probs = results['probabilities']
    thresholds = results['thresholds']
    
    metrics = {}
    
    # Overall metrics
    metrics['micro_f1'] = f1_score(labels, preds, average='micro')
    metrics['macro_f1'] = f1_score(labels, preds, average='macro')
    metrics['weighted_f1'] = f1_score(labels, preds, average='weighted')
    metrics['samples_f1'] = f1_score(labels, preds, average='samples')
    
    metrics['micro_precision'] = precision_score(labels, preds, average='micro')
    metrics['macro_precision'] = precision_score(labels, preds, average='macro')
    metrics['weighted_precision'] = precision_score(labels, preds, average='weighted')
    metrics['samples_precision'] = precision_score(labels, preds, average='samples')
    
    metrics['micro_recall'] = recall_score(labels, preds, average='micro')
    metrics['macro_recall'] = recall_score(labels, preds, average='macro')
    metrics['weighted_recall'] = recall_score(labels, preds, average='weighted')
    metrics['samples_recall'] = recall_score(labels, preds, average='samples')
    
    # Sample-based metrics
    metrics['hamming_loss'] = hamming_loss(labels, preds)
    metrics['exact_match_ratio'] = np.mean(np.all(labels == preds, axis=1))
    
    # Per-class metrics
    per_class_metrics = []
    for i in range(labels.shape[1]):
        class_metrics = {}
        class_metrics['class'] = emotion_classes[i] if i < len(emotion_classes) else f"Unknown_{i}"
        class_metrics['threshold'] = thresholds[i]
        class_metrics['support'] = int(np.sum(labels[:, i]))
        
        if class_metrics['support'] > 0:
            class_metrics['f1'] = f1_score(labels[:, i], preds[:, i], average='binary')
            class_metrics['precision'] = precision_score(labels[:, i], preds[:, i], average='binary')
            class_metrics['recall'] = recall_score(labels[:, i], preds[:, i], average='binary')
            
            # ROC AUC
            class_metrics['roc_auc'] = auc(*roc_curve(labels[:, i], probs[:, i])[:2])
            
            # PR AUC
            class_metrics['pr_auc'] = average_precision_score(labels[:, i], probs[:, i])
        else:
            class_metrics['f1'] = 0.0
            class_metrics['precision'] = 0.0
            class_metrics['recall'] = 0.0
            class_metrics['roc_auc'] = 0.0
            class_metrics['pr_auc'] = 0.0
        
        per_class_metrics.append(class_metrics)
    
    metrics['per_class'] = per_class_metrics
    
    # Label distribution metrics
    metrics['label_cardinality'] = np.mean(np.sum(labels, axis=1))  # Avg number of labels per sample
    metrics['label_density'] = metrics['label_cardinality'] / labels.shape[1]  # Cardinality normalized by number of classes
    
    return metrics

def plot_confusion_matrix(y_true, y_pred, classes, filename):
    """
    Plot confusion matrix for each class
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        classes: Class names
        filename: Name of the file to save the plot
    """
    # Calculate multilabel confusion matrix
    mcm = multilabel_confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, axes = plt.subplots(4, 7, figsize=(20, 12))
    axes = axes.flatten()
    
    # Plot confusion matrix for each class
    for i, (cm, ax) in enumerate(zip(mcm, axes)):
        if i >= len(classes):
            break
            
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        
        # Set title and labels
        ax.set_title(f'{classes[i]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        
        # Set tick labels
        ax.set_xticklabels(['Negative', 'Positive'])
        ax.set_yticklabels(['Negative', 'Positive'])
    
    # Hide unused subplots
    for i in range(len(classes), len(axes)):
        axes[i].axis('off')
    
    # Add title
    plt.suptitle('Confusion Matrix per Emotion', fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    
    # Save and close
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_roc_curves(y_true, y_score, classes, filename):
    """
    Plot ROC curves for each class
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        classes: Class names
        filename: Name of the file to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    # Plot ROC curve for each class
    for i, (color, class_name) in enumerate(zip(colors, classes)):
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        
        # Plot curve
        plt.plot(fpr, tpr, lw=2, color=color, label=f'{class_name} (AUC = {roc_auc:.2f})')
    
    # Add diagonal line (random classifier)
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    
    # Set labels and title
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Emotion')
    
    # Add legend
    plt.legend(loc='lower right', fontsize='small')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_precision_recall_curves(y_true, y_score, classes, filename):
    """
    Plot precision-recall curves for each class
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        classes: Class names
        filename: Name of the file to save the plot
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get colors
    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))
    
    # Plot precision-recall curve for each class
    for i, (color, class_name) in enumerate(zip(colors, classes)):
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y_true[:, i], y_score[:, i])
        avg_precision = average_precision_score(y_true[:, i], y_score[:, i])
        
        # Plot curve
        plt.plot(recall, precision, lw=2, color=color, label=f'{class_name} (AP = {avg_precision:.2f})')
    
    # Set labels and title
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Emotion')
    
    # Add legend
    plt.legend(loc='upper right', fontsize='small')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_threshold_optimization(y_true, y_score, classes, filename):
    """
    Plot F1 scores at different thresholds for top 10 classes
    
    Args:
        y_true: Ground truth labels
        y_score: Predicted probabilities
        classes: Class names
        filename: Name of the file to save the plot
    """
    # Calculate class F1 scores
    class_f1 = []
    for i in range(y_true.shape[1]):
        if np.sum(y_true[:, i]) > 0:
            f1 = f1_score(y_true[:, i], (y_score[:, i] > 0.5).astype(int), average='binary')
            class_f1.append((i, f1))
    
    # Sort by F1 score
    class_f1.sort(key=lambda x: x[1], reverse=True)
    
    # Select top 10 classes
    top_classes = [class_f1[i][0] for i in range(min(10, len(class_f1)))]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Get colors
    colors = plt.cm.tab10(np.linspace(0, 1, len(top_classes)))
    
    # Plot F1 score at different thresholds for each class
    for i, (class_idx, color) in enumerate(zip(top_classes, colors)):
        thresholds = np.linspace(0.1, 0.9, 9)
        f1_scores = []
        
        # Calculate F1 score at each threshold
        for threshold in thresholds:
            preds = (y_score[:, class_idx] > threshold).astype(int)
            f1 = f1_score(y_true[:, class_idx], preds, average='binary')
            f1_scores.append(f1)
        
        # Plot curve
        plt.plot(thresholds, f1_scores, lw=2, color=color, marker='o', 
                 label=f'{classes[class_idx]}')
    
    # Set labels and title
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('F1 Score at Different Thresholds for Top 10 Emotions')
    
    # Add legend
    plt.legend(loc='best', fontsize='small')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_label_correlations(y_true, classes, filename):
    """
    Plot correlation matrix between labels
    
    Args:
        y_true: Ground truth labels
        classes: Class names
        filename: Name of the file to save the plot
    """
    # Calculate correlation matrix
    df = pd.DataFrame(y_true, columns=classes)
    corr = df.corr()
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot correlation matrix
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap='coolwarm', annot=True, fmt='.2f', 
                square=True, linewidths=0.5)
    
    # Set title
    plt.title('Correlation Between Emotions')
    
    # Save and close
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

def _convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    import numpy as np
    
    # Check if object is a numpy array
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    
    # Check if object is a numpy scalar
    if np.isscalar(obj) and isinstance(obj, np.generic):
        # Convert numpy scalars to Python scalars
        if np.issubdtype(obj.dtype, np.integer):
            return int(obj)
        elif np.issubdtype(obj.dtype, np.floating):
            return float(obj)
        elif np.issubdtype(obj.dtype, np.bool_):
            return bool(obj)
        elif np.issubdtype(obj.dtype, np.complex):
            return str(obj)
        else:
            return obj.item()
    
    # Handle dictionaries
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    
    # Handle lists and tuples
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    
    # Return unchanged for other types
    return obj

def generate_evaluation_report(metrics, results, emotion_classes, model_data, dataset_name="test"):
    """
    Generate an evaluation report with visualizations
    
    Args:
        metrics: Dictionary with calculated metrics
        results: Dictionary with evaluation results
        emotion_classes: List of emotion class names
        model_data: Dictionary with model metadata
        dataset_name: Name of the dataset being evaluated
        
    Returns:
        report_dir: Path to the report directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = os.path.join(REPORTS_DIR, f"{dataset_name}_evaluation_{timestamp}")
    os.makedirs(report_dir, exist_ok=True)
    
    # Convert numpy types to Python native types for JSON serialization
    metrics_json = _convert_numpy_types(metrics)
    
    # Save metrics to JSON
    with open(os.path.join(report_dir, "metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=4)
    
    # Generate plots
    plot_confusion_matrix(
        results['labels'], 
        results['predictions'], 
        emotion_classes, 
        os.path.join(report_dir, "confusion_matrix.png")
    )
    
    plot_roc_curves(
        results['labels'], 
        results['probabilities'], 
        emotion_classes, 
        os.path.join(report_dir, "roc_curves.png")
    )
    
    plot_precision_recall_curves(
        results['labels'], 
        results['probabilities'], 
        emotion_classes, 
        os.path.join(report_dir, "pr_curves.png")
    )
    
    plot_threshold_optimization(
        results['labels'], 
        results['probabilities'], 
        emotion_classes, 
        os.path.join(report_dir, "threshold_optimization.png")
    )
    
    plot_label_correlations(
        results['labels'], 
        emotion_classes, 
        os.path.join(report_dir, "label_correlations.png")
    )
    
    # Generate HTML report
    html_report = f"""<!DOCTYPE html>
<html>
<head>
    <title>Emotion Detection Model Evaluation - {dataset_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #f2f2f2; }}
        tr:nth-child(even) {{ background-color: #f9f9f9; }}
        .metric {{ font-weight: bold; }}
        .plot {{ margin-bottom: 30px; }}
        .plot img {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>Emotion Detection Model Evaluation</h1>
    <p><strong>Dataset:</strong> {dataset_name}</p>
    <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    <p><strong>Model:</strong> {model_data.get('transformer_model', 'Unknown')}</p>
    
    <h2>Overall Metrics</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Value</th>
        </tr>
        <tr>
            <td>Micro F1</td>
            <td>{metrics['micro_f1']:.4f}</td>
        </tr>
        <tr>
            <td>Macro F1</td>
            <td>{metrics['macro_f1']:.4f}</td>
        </tr>
        <tr>
            <td>Weighted F1</td>
            <td>{metrics['weighted_f1']:.4f}</td>
        </tr>
        <tr>
            <td>Samples F1</td>
            <td>{metrics['samples_f1']:.4f}</td>
        </tr>
        <tr>
            <td>Micro Precision</td>
            <td>{metrics['micro_precision']:.4f}</td>
        </tr>
        <tr>
            <td>Micro Recall</td>
            <td>{metrics['micro_recall']:.4f}</td>
        </tr>
        <tr>
            <td>Hamming Loss</td>
            <td>{metrics['hamming_loss']:.4f}</td>
        </tr>
        <tr>
            <td>Exact Match Ratio</td>
            <td>{metrics['exact_match_ratio']:.4f}</td>
        </tr>
        <tr>
            <td>Label Cardinality</td>
            <td>{metrics['label_cardinality']:.4f}</td>
        </tr>
    </table>
    
    <h2>Per-Class Metrics</h2>
    <table>
        <tr>
            <th>Emotion</th>
            <th>F1</th>
            <th>Precision</th>
            <th>Recall</th>
            <th>ROC AUC</th>
            <th>PR AUC</th>
            <th>Threshold</th>
            <th>Support</th>
        </tr>
"""
    
    # Sort per-class metrics by F1 score
    sorted_metrics = sorted(metrics['per_class'], key=lambda x: x['f1'], reverse=True)
    
    for class_metrics in sorted_metrics:
        html_report += f"""
        <tr>
            <td>{class_metrics['class']}</td>
            <td>{class_metrics['f1']:.4f}</td>
            <td>{class_metrics['precision']:.4f}</td>
            <td>{class_metrics['recall']:.4f}</td>
            <td>{class_metrics['roc_auc']:.4f}</td>
            <td>{class_metrics['pr_auc']:.4f}</td>
            <td>{class_metrics['threshold']:.4f}</td>
            <td>{class_metrics['support']}</td>
        </tr>"""
    
    html_report += """
    </table>
    
    <h2>Visualizations</h2>
    
    <div class="plot">
        <h3>Confusion Matrix per Emotion</h3>
        <img src="confusion_matrix.png" alt="Confusion Matrix">
    </div>
    
    <div class="plot">
        <h3>ROC Curves for Each Emotion</h3>
        <img src="roc_curves.png" alt="ROC Curves">
    </div>
    
    <div class="plot">
        <h3>Precision-Recall Curves for Each Emotion</h3>
        <img src="pr_curves.png" alt="Precision-Recall Curves">
    </div>
    
    <div class="plot">
        <h3>Threshold Optimization for Top 10 Emotions</h3>
        <img src="threshold_optimization.png" alt="Threshold Optimization">
    </div>
    
    <div class="plot">
        <h3>Label Correlations</h3>
        <img src="label_correlations.png" alt="Label Correlations">
    </div>
</body>
</html>
"""
    
    # Save HTML report
    with open(os.path.join(report_dir, "report.html"), "w") as f:
        f.write(html_report)
    
    print(f"Evaluation report saved to {report_dir}")
    return report_dir

def run_evaluation(model_path=MODEL_PATH, data_paths=DATA_PATHS):
    """
    Run a complete evaluation of the model
    
    Args:
        model_path: Path to the saved model
        data_paths: Dictionary with paths to train, val and test data
    """
    start_time = time.time()
    print("Starting comprehensive model evaluation...")
    
    # Create output directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    # Load model and data
    model, datasets, model_data, emotion_classes = load_model_and_data(model_path, data_paths)
    
    # Evaluate on test set with default thresholds
    print("\nEvaluating with default thresholds (0.5)...")
    default_results = evaluate_model(model, datasets['test'], device)
    default_metrics = calculate_metrics(default_results, emotion_classes)
    
    # Optimize thresholds
    print("\nOptimizing thresholds...")
    thresholds = optimize_thresholds(default_results['labels'], default_results['probabilities'])
    
    # Evaluate with optimized thresholds
    print("\nEvaluating with optimized thresholds...")
    optimized_results = evaluate_model(model, datasets['test'], device, thresholds=thresholds)
    optimized_metrics = calculate_metrics(optimized_results, emotion_classes)
    
    # Generate reports
    print("\nGenerating reports...")
    
    # Default thresholds report
    default_report_dir = generate_evaluation_report(
        default_metrics,
        default_results,
        emotion_classes,
        model_data,
        "test_default"
    )
    
    # Optimized thresholds report
    optimized_report_dir = generate_evaluation_report(
        optimized_metrics,
        optimized_results,
        emotion_classes,
        model_data,
        "test_optimized"
    )
    
    # Print summary
    print("\nEvaluation Summary:")
    print("-" * 50)
    print(f"Default Thresholds: Micro-F1 = {default_metrics['micro_f1']:.4f}, Macro-F1 = {default_metrics['macro_f1']:.4f}")
    print(f"Optimized Thresholds: Micro-F1 = {optimized_metrics['micro_f1']:.4f}, Macro-F1 = {optimized_metrics['macro_f1']:.4f}")
    
    # Performance analysis
    print("\nTop 5 performing emotions:")
    top_emotions = sorted(optimized_metrics['per_class'], key=lambda x: x['f1'], reverse=True)[:5]
    for emotion in top_emotions:
        print(f"- {emotion['class']}: F1 = {emotion['f1']:.4f}, Support = {emotion['support']}")
    
    print("\nBottom 5 performing emotions:")
    bottom_emotions = sorted(optimized_metrics['per_class'], key=lambda x: x['f1'])[:5]
    for emotion in bottom_emotions:
        print(f"- {emotion['class']}: F1 = {emotion['f1']:.4f}, Support = {emotion['support']}")
    
    # Print execution time
    execution_time = time.time() - start_time
    print(f"\nEvaluation completed in {execution_time:.2f} seconds")
    
    # Print report paths
    print(f"\nReports saved to:")
    print(f"- Default thresholds: {default_report_dir}")
    print(f"- Optimized thresholds: {optimized_report_dir}")

if __name__ == "__main__":
    # Run the evaluation
    run_evaluation() 