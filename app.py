from flask import Flask, render_template, request, jsonify
import os
import json
import torch
from model import predict_emotions
import plotly.express as px
import pandas as pd
import uuid
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'emotion-detection-secret'
app.config['CONVERSATIONS'] = {}  # Store conversations in memory

# Ensure required directories exist
os.makedirs('static', exist_ok=True)
os.makedirs('static/images', exist_ok=True)
os.makedirs('static/js', exist_ok=True)
os.makedirs('static/css', exist_ok=True)

# Load emotion mapping for visualization
with open('emotion_mapping.json', 'r') as f:
    EMOTION_MAPPING = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    text = request.form.get('text', '')
    if not text:
        return jsonify({'error': 'No text provided'})
    
    try:
        # Predict emotions for the text
        emotions = predict_emotions(text)
        
        # Sort emotions by probability
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1]['probability'], reverse=True)
        
        # Return the results
        return jsonify({
            'text': text,
            'emotions': emotions,
            'top_emotions': sorted_emotions[:5]  # Top 5 emotions
        })
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/start_conversation', methods=['POST'])
def start_conversation():
    # Generate a unique ID for the conversation
    conv_id = str(uuid.uuid4())
    app.config['CONVERSATIONS'][conv_id] = {
        'texts': [],
        'timestamps': [],
        'emotions': []
    }
    return jsonify({'conversation_id': conv_id})

@app.route('/add_to_conversation', methods=['POST'])
def add_to_conversation():
    data = request.json
    conv_id = data.get('conversation_id')
    text = data.get('text')
    
    if not conv_id or not text or conv_id not in app.config['CONVERSATIONS']:
        return jsonify({'error': 'Invalid conversation ID or text'})
    
    try:
        # Predict emotions for the text
        emotions = predict_emotions(text)
        
        # Add to conversation history
        app.config['CONVERSATIONS'][conv_id]['texts'].append(text)
        app.config['CONVERSATIONS'][conv_id]['timestamps'].append(datetime.now().strftime('%H:%M:%S'))
        app.config['CONVERSATIONS'][conv_id]['emotions'].append(emotions)
        
        # Generate updated emotion progression graph
        graph_data = generate_emotion_graph(conv_id)
        
        return jsonify({
            'text': text,
            'emotions': emotions,
            'graph_data': graph_data
        })
    except Exception as e:
        return jsonify({'error': str(e)})

def generate_emotion_graph(conv_id):
    conversation = app.config['CONVERSATIONS'][conv_id]
    
    if not conversation['texts']:
        return {}
    
    # Extract emotion data for visualization
    data = []
    all_emotions = set()  # Track all emotions that appear in the conversation
    
    # First pass: collect all unique emotions
    for emotion_dict in conversation['emotions']:
        for emotion, details in emotion_dict.items():
            if details['detected']:
                all_emotions.add(emotion)
    
    # Second pass: create data points for each emotion across all messages
    for i, emotions_dict in enumerate(conversation['emotions']):
        message_id = i+1
        timestamp = conversation['timestamps'][i]
        text = conversation['texts'][i][:30] + '...' if len(conversation['texts'][i]) > 30 else conversation['texts'][i]
        
        # Add data for all tracked emotions, even if not present in this message
        for emotion in all_emotions:
            if emotion in emotions_dict:
                data.append({
                    'message_id': message_id,
                    'text': text,
                    'timestamp': timestamp,
                    'emotion': emotion,
                    'probability': emotions_dict[emotion]['probability']
                })
            else:
                # Add with zero probability if emotion not in this message
                data.append({
                    'message_id': message_id,
                    'text': text,
                    'timestamp': timestamp,
                    'emotion': emotion,
                    'probability': 0
                })
    
    if not data:
        return {}
    
    # Convert to dataframe for easier plotting
    df = pd.DataFrame(data)
    
    # Get top emotions across the conversation (based on average probability)
    top_emotions = df.groupby('emotion')['probability'].mean().sort_values(ascending=False).head(5).index.tolist()
    
    # Prepare graph data to return to frontend
    graph_data = {
        'message_ids': df['message_id'].tolist(),
        'timestamps': df['timestamp'].tolist(),
        'emotions': df['emotion'].tolist(),
        'probabilities': df['probability'].tolist(),
        'top_emotions': top_emotions,
        'all_emotions': list(all_emotions)  # Send all detected emotions to frontend
    }
    
    return graph_data

if __name__ == '__main__':
    app.run(debug=True, port=5000) 