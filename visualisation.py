import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from model import predict_emotions
import torch.serialization
import numpy

# Create visualisations directory
os.makedirs("visualisations", exist_ok=True)

# For PyTorch 2.6+ compatibility
try:
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except:
    pass

# Set style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.2)

def create_emotion_bar_chart(emotions, title, filename="emotion_bar_chart.png"):
    """Create a bar chart of emotion probabilities"""
    # Sort emotions by probability
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    # Extract emotions and probabilities
    emotion_names = [e for e, _ in sorted_emotions[:10]]  # Top 10 emotions
    probs = [d['probability'] for _, d in sorted_emotions[:10]]
    thresholds = [d['threshold'] for _, d in sorted_emotions[:10]]
    is_detected = [d['detected'] for _, d in sorted_emotions[:10]]
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create bars
    bars = plt.bar(emotion_names, probs, color='lightblue', alpha=0.7)
    
    # Add threshold lines
    for i, threshold in enumerate(thresholds):
        plt.plot([i-0.4, i+0.4], [threshold, threshold], color='red', linestyle='--', linewidth=1)
    
    # Highlight detected emotions
    for i, detected in enumerate(is_detected):
        if detected:
            bars[i].set_color('darkblue')
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.ylabel('Probability', fontsize=14)
    plt.ylim(0, max(probs) * 1.2)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Add a legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='darkblue', label='Detected'),
        Patch(facecolor='lightblue', alpha=0.7, label='Not Detected'),
        plt.Line2D([0], [0], color='red', linestyle='--', label='Threshold')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    # Save figure
    plt.savefig(os.path.join("visualisations", filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def create_radar_chart(emotions, title, filename="emotion_radar_chart.png"):
    """Create a radar/spider chart for multiple emotions"""
    # Sort emotions by probability
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    # Get top emotions
    top_emotions = sorted_emotions[:8]  # Top 8 emotions for radar
    
    # Extract data
    emotion_names = [e for e, _ in top_emotions]
    probs = [d['probability'] for _, d in top_emotions]
    
    # Number of variables
    N = len(emotion_names)
    
    # Create angles for each emotion (evenly spaced around the circle)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the loop
    
    # Add probabilities (and close the loop)
    probs += probs[:1]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    
    # Draw the chart
    ax.plot(angles, probs, linewidth=2, linestyle='solid', color='blue')
    ax.fill(angles, probs, color='blue', alpha=0.4)
    
    # Set ticks and labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(emotion_names)
    
    # Set y-limits
    ax.set_ylim(0, max(probs) * 1.2)
    
    # Add title
    plt.title(title, fontsize=16, y=1.1)
    
    # Save figure
    plt.savefig(os.path.join("visualisations", filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def visualise_threshold_impact(emotions, title, filename="threshold_impact.png"):
    """Visualise how different thresholds affect detection"""
    # Sort emotions by probability
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1]['probability'], reverse=True)
    
    # Extract top emotions
    top_emotions = sorted_emotions[:10]
    
    # Extract data
    emotion_names = [e for e, _ in top_emotions]
    probs = [d['probability'] for _, d in top_emotions]
    current_thresholds = [d['threshold'] for _, d in top_emotions]
    
    # Create range of thresholds
    threshold_range = np.linspace(0, 1, 11)
    
    # Create matrix of detected emotions at each threshold
    detection_matrix = np.zeros((len(top_emotions), len(threshold_range)))
    for i, prob in enumerate(probs):
        for j, thresh in enumerate(threshold_range):
            detection_matrix[i, j] = 1 if prob >= thresh else 0
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Create heatmap
    sns.heatmap(detection_matrix, 
                cmap=["white", "green"],
                cbar=False,
                yticklabels=emotion_names,
                xticklabels=[f"{t:.1f}" for t in threshold_range],
                linewidths=1)
    
    # Mark current thresholds
    for i, threshold in enumerate(current_thresholds):
        # Find closest threshold in our range
        closest_idx = np.abs(threshold_range - threshold).argmin()
        plt.scatter(closest_idx + 0.5, i + 0.5, color='red', s=100, marker='x')
    
    # Customize plot
    plt.title(title, fontsize=16)
    plt.xlabel('Threshold Value', fontsize=14)
    plt.ylabel('Emotion', fontsize=14)
    
    # Add legend for current threshold
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='x', color='w', markerfacecolor='red', 
                              markersize=12, label='Current Threshold')]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(os.path.join("visualisations", filename), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")

def analyze_text_and_visualise(text):
    """Analyze a text and create visualisations for it"""
    print(f"Analyzing: '{text}'")
    
    # Get emotion predictions
    emotions = predict_emotions(text)
    
    # Clean the text for filename
    clean_text = text.replace(" ", "_").replace(".", "").replace(",", "")[:30]
    
    # Create visualisations
    create_emotion_bar_chart(
        emotions, 
        f"Emotion Analysis: '{text}'",
        f"bar_chart_{clean_text}.png"
    )
    
    create_radar_chart(
        emotions, 
        f"Emotion Radar: '{text}'",
        f"radar_chart_{clean_text}.png"
    )
    
    visualise_threshold_impact(
        emotions, 
        f"Threshold Impact: '{text}'",
        f"threshold_impact_{clean_text}.png"
    )
    
    print(f"Created visualisations for: '{text}'\n")

def visualise_batch_predictions(texts, filename_prefix="batch"):
    """Visualise predictions for a batch of texts"""
    # Get predictions for all texts
    all_predictions = {}
    for i, text in enumerate(texts):
        all_predictions[f"Text {i+1}"] = predict_emotions(text)
    
    # Create combined bar chart
    plt.figure(figsize=(14, 10))
    
    # Get all unique emotions across all texts
    all_emotions = set()
    for pred in all_predictions.values():
        all_emotions.update(pred.keys())
    
    # Sort emotions alphabetically
    all_emotions = sorted(all_emotions)
    
    # Create dataframe for easier plotting
    df = pd.DataFrame(index=all_emotions, columns=all_predictions.keys())
    
    # Fill dataframe with probabilities
    for text_name, pred in all_predictions.items():
        for emotion in all_emotions:
            if emotion in pred:
                df.loc[emotion, text_name] = pred[emotion]['probability']
            else:
                df.loc[emotion, text_name] = 0
    
    # Plot
    df.plot(kind='bar', figsize=(14, 10), width=0.8)
    plt.title("Emotion Comparison Across Texts", fontsize=16)
    plt.xlabel("Emotion", fontsize=14)
    plt.ylabel("Probability", fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.legend(title="")
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join("visualisations", f"{filename_prefix}_comparison.png"), 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename_prefix}_comparison.png")

if __name__ == "__main__":
    # Example texts to analyze
    texts = [
        "I'm so excited about this new project, although I'm a bit nervous too.",
        "This makes me angry and frustrated. I can't believe they did that.",
        "Thank you so much for your help. I really appreciate it.",
    ]
    
    # Create individual visualisations for each text
    for text in texts:
        analyze_text_and_visualise(text)
    
    # Create batch visualization comparing all texts
    visualise_batch_predictions(texts, "emotion_comparison")
    
    print("All visualisations created successfully in 'visualisations' directory!") 