from model import predict_emotions
import torch.serialization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime

# For PyTorch 2.6+ compatibility
try:
    torch.serialization.add_safe_globals([numpy._core.multiarray._reconstruct])
except:
    pass

# Ensure testing directory exists
os.makedirs("testing", exist_ok=True)

def analyze_emotion_progression(texts, title="Emotion Progression Analysis"):
    """
    Analyze emotions in a sequence of texts and track changes over time
    
    Args:
        texts: List of texts in chronological order
        title: Title for the analysis
    
    Returns:
        DataFrame with emotion data
    """
    # Create timestamp for output files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n{'='*60}")
    print(f"ANALYZING EMOTION PROGRESSION: {title}")
    print(f"{'='*60}")
    
    results = []
    
    # Analyze each text
    for idx, text in enumerate(texts):
        print(f"\nText {idx+1}:")
        print(f"- Content: {text}")
        
        # Predict emotions
        emotions = predict_emotions(text)
        
        # Store detected emotions
        detected = []
        for emotion, data in emotions.items():
            if data['detected']:
                detected.append(f"{emotion} ({data['probability']:.3f})")
        
        print(f"- Detected: {', '.join(detected) if detected else 'None'}")
        
        # Store top 3 emotions
        top_emotions = list(emotions.items())[:3]
        print("- Top 3 emotions:")
        for emotion, data in top_emotions:
            print(f"  • {emotion}: {data['probability']:.3f}")
        
        # Store result for each emotion
        for emotion, data in emotions.items():
            results.append({
                'text_idx': idx + 1,
                'text': text,
                'emotion': emotion,
                'probability': data['probability'],
                'detected': data['detected']
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Create output directory
    output_dir = f"testing/emotion_progression_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save raw data as CSV
    df.to_csv(f"{output_dir}/emotion_data.csv", index=False)
    
    # Save original texts
    with open(f"{output_dir}/texts.json", 'w') as f:
        json.dump([{"text_idx": i+1, "text": t} for i, t in enumerate(texts)], f, indent=2)
    
    # Generate visualizations
    generate_visualizations(df, output_dir, title)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return df

def generate_visualizations(df, output_dir, title):
    """Generate visualizations from emotion data"""
    # Set plot style
    sns.set_style("whitegrid")
    plt.rcParams.update({'font.size': 12})
    
    # 1. Line plot showing emotion changes over time
    plt.figure(figsize=(14, 8))
    
    # Filter for top 5 emotions only for clarity
    top_emotions = df.groupby('emotion')['probability'].mean().sort_values(ascending=False).head(5).index.tolist()
    filtered_df = df[df['emotion'].isin(top_emotions)]
    
    # Pivot data for plotting
    pivot_data = filtered_df.pivot(index='text_idx', columns='emotion', values='probability')
    
    # Plot each emotion
    for emotion in pivot_data.columns:
        pivot_data[emotion].plot(
            marker='o', 
            linestyle='-', 
            label=f"{emotion}"
        )
    
    plt.title(f"Emotion Progression Over Time\n{title}")
    plt.xlabel("Text Number")
    plt.ylabel("Emotion Probability")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/emotion_progression.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Heatmap of emotions over time
    plt.figure(figsize=(12, 8))
    
    # Create pivot table: texts x emotions with probability values
    pivot_data = df.pivot(index='text_idx', columns='emotion', values='probability')
    
    # Sort emotions by mean value
    emotion_order = pivot_data.mean().sort_values(ascending=False).index
    pivot_data = pivot_data[emotion_order]
    
    # Keep only top 10 emotions
    pivot_data = pivot_data.iloc[:, :10]
    
    # Create heatmap
    sns.heatmap(pivot_data, annot=True, cmap="YlOrRd", fmt=".2f", linewidths=.5)
    plt.title(f"Emotion Heatmap\n{title}")
    plt.ylabel("Text Number")
    plt.xlabel("Emotion")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/emotion_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

# Example usage
if __name__ == "__main__":
    # Example sequence of social media posts or messages
    texts = [
        "I'm so excited about starting this new project today. Can't wait to see what we'll accomplish!",
        "Hmm, running into some unexpected issues with the implementation. This might be trickier than I thought.",
        "Spent hours debugging but still can't figure out what's wrong. This is so frustrating!",
        "Finally found the bug! It was a simple typo all along. Feeling relieved but also a bit silly.",
        "Made great progress today after fixing that bug. The project is starting to take shape.",
        "Just presented the project to the team and everyone loved it! All the hard work was worth it.",
    ]
    
    # Analyze the emotion progression
    analyze_emotion_progression(texts, "Project Development Journey")
    
    # Instructions for custom analysis
    print("\n\nYou can analyze your own text sequence by creating a list of texts")
    print("and calling analyze_emotion_progression(your_texts, 'Your Title')")
    print("\nExample:")
    print("texts = [")
    print("    \"I'm feeling really anxious about the upcoming exam.\",")
    print("    \"I've been studying all day and I'm getting more confident.\",")
    print("    \"The exam went well! I think I'll get a good grade.\",")
    print("]")
    print("analyze_emotion_progression(texts, \"Exam Preparation Journey\")")