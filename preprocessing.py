import kagglehub
import os
import re
import pandas as pd
import numpy as np
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pickle
import time

# Start timing the script
start_time = time.time()

print("="*80)
print("SOCIAL MEDIA SENTIMENT ANALYSIS - DATA PREPROCESSING PIPELINE")
print("="*80)

# Fix SSL certificate issue
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK data files with SSL verification disabled
print("\n1. DOWNLOADING NLTK RESOURCES")
print("-"*50)
try:
    # Create NLTK data directory if it doesn't exist
    nltk_data_dir = os.path.expanduser('~/nltk_data')
    os.makedirs(nltk_data_dir, exist_ok=True)
    os.makedirs(os.path.join(nltk_data_dir, 'corpora'), exist_ok=True)
    os.makedirs(os.path.join(nltk_data_dir, 'tokenizers'), exist_ok=True)
    
    nltk.download('punkt', quiet=False)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    print("✓ NLTK resources downloaded successfully")
    
    # Verify that punkt was downloaded
    punkt_path = os.path.join(nltk_data_dir, 'tokenizers', 'punkt')
    if os.path.exists(punkt_path):
        print("✓ Punkt tokenizer verified")
    else:
        print("! Punkt tokenizer not found, will use simple tokenizer")
except Exception as e:
    print(f"! Error downloading NLTK data: {e}")
    print("  Continuing with fallback methods")

# Check if tokenizer is available
tokenizer_available = True
try:
    # Try tokenizing a simple test sentence
    test_tokens = word_tokenize("This is a test sentence.")
    print(f"✓ NLTK tokenizer working (test: {test_tokens[:3]}...)")
except Exception as e:
    tokenizer_available = False
    print(f"! NLTK tokenizer not working: {e}")

# Initialize NLP components
try:
    stop_words = set(stopwords.words('english'))
    print("✓ Using NLTK stopwords")
except LookupError:
    print("! Using minimal stopwords set (fallback)")
    # Minimal set of English stopwords
    stop_words = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'because', 'as', 'what',
                  'which', 'this', 'that', 'these', 'those', 'then', 'just', 'so', 'than',
                  'such', 'both', 'through', 'about', 'for', 'is', 'of', 'while', 'during',
                  'to', 'from', 'in', 'on', 'at', 'by', 'with', 'about', 'against',
                  'between', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'up', 'down', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                  'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                  'i', 'me', 'my', 'myself', 'you', 'your', 'yourself', 'he', 'him', 'his',
                  'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they',
                  'them', 'their', 'theirs', 'themselves', 'we', 'us', 'our', 'ours',
                  'ourselves', 'who', 'whom', 'whose', 'where', 'when', 'why', 'how'}

try:
    lemmatizer = WordNetLemmatizer()
    print("✓ Using WordNetLemmatizer")
except LookupError:
    print("! Using identity lemmatizer (fallback)")
    class IdentityLemmatizer:
        def lemmatize(self, word):
            return word
    lemmatizer = IdentityLemmatizer()

try:
    stemmer = PorterStemmer()
    print("✓ Using PorterStemmer")
except Exception as e:
    print(f"! Error initializing stemmer: {e}")
    class IdentityStemmer:
        def stem(self, word):
            return word
    stemmer = IdentityStemmer()

# Define tokenization functions
def simple_tokenize(text):
    """Simple tokenizer that splits on whitespace and punctuation"""
    if not isinstance(text, str):
        return []
    text = re.sub(r'[^\w\s]', ' ', text.lower())
    return text.split()

def tokenize_text(text):
    if not isinstance(text, str):
        return []
    
    if tokenizer_available:
        try:
            return word_tokenize(text)
        except Exception:
            return simple_tokenize(text)
    else:
        return simple_tokenize(text)

# Define text normalization patterns
url_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
mention_pattern = r"@\w+"
hashtag_pattern = r"#\w+"
special_char_pattern = r"[^A-Za-z0-9\s]"
extra_spaces_pattern = r"\s+"
numbers_pattern = r"\b\d+\b"

# Create output directory
os.makedirs("processed_data", exist_ok=True)

# Download and prepare dataset
print("\n2. DATA COLLECTION")
print("-"*50)
try:
    path = kagglehub.dataset_download("debarshichanda/goemotions")
    print(f"✓ Dataset downloaded to: {path}")
    
    # Define paths to the dataset files
    train_file = os.path.join(path, "data/train.tsv")
    dev_file = os.path.join(path, "data/dev.tsv")
    test_file = os.path.join(path, "data/test.tsv")
    
    # Load all data files
    print("\nLoading dataset files...")
    train_df = pd.read_csv(train_file, sep='\t', header=None, 
                         names=['text', 'emotion_id', 'comment_id'])
    dev_df = pd.read_csv(dev_file, sep='\t', header=None, 
                       names=['text', 'emotion_id', 'comment_id'])
    test_df = pd.read_csv(test_file, sep='\t', header=None, 
                        names=['text', 'emotion_id', 'comment_id'])
    
    # Combine all data for preprocessing
    print("Combining dataset splits...")
    df = pd.concat([train_df, dev_df, test_df], ignore_index=True)
    
    # Limit dataset size if it's too large
    if len(df) > 20000:  # Limit to 20k entries for faster processing
        df = df.sample(20000, random_state=42)
        print(f"  Limiting to {len(df)} entries for manageable processing")
    
    print(f"✓ Combined dataset with {len(df)} entries")
    
    # Create a mapping for emotion IDs if available
    try:
        emotions_file = os.path.join(path, "data/emotions.txt")
        with open(emotions_file, 'r') as f:
            emotion_labels = [line.strip() for line in f.readlines()]
        emotion_map = {i: label for i, label in enumerate(emotion_labels)}
        df['emotion'] = df['emotion_id'].map(emotion_map)
        print(f"✓ Mapped {len(emotion_map)} emotion labels")
    except Exception as e:
        print(f"! Could not load emotion mappings: {e}")
    
except Exception as e:
    print(f"! Error loading dataset: {e}")
    print("Creating sample dataset for demonstration...")
    # Create a small sample dataframe
    df = pd.DataFrame({
        'text': [
            "I'm feeling really happy today!",
            "This makes me so angry and frustrated.",
            "I'm sad about what happened yesterday.",
            "I'm anxious about the upcoming exam.",
            "Just feeling neutral about everything right now."
        ],
        'emotion_id': [0, 1, 2, 3, 4],
        'comment_id': ['id1', 'id2', 'id3', 'id4', 'id5'],
        'emotion': ['joy', 'anger', 'sadness', 'anxiety', 'neutral']
    })

# Data cleaning and preprocessing
print("\n3. DATA CLEANING")
print("-"*50)

# Remove duplicate entries
original_size = len(df)
df = df.drop_duplicates()
print(f"✓ Removed {original_size - len(df)} duplicate entries")

# Define comprehensive text cleaning function
def clean_text(text):
    """
    Comprehensive text cleaning function that:
    - Removes URLs, mentions, hashtags
    - Removes special characters and extra spaces
    - Converts text to lowercase
    """
    if not isinstance(text, str):
        return ""
    
    # Remove URLs, mentions, hashtags
    text = re.sub(url_pattern, "", text)
    text = re.sub(mention_pattern, "", text)
    text = re.sub(hashtag_pattern, "", text)
    
    # Remove or normalize numbers
    text = re.sub(numbers_pattern, " NUM ", text)
    
    # Remove special characters
    text = re.sub(special_char_pattern, " ", text)
    
    # Remove extra spaces and convert to lowercase
    text = re.sub(extra_spaces_pattern, " ", text)
    text = text.lower().strip()
    
    return text

# Apply text cleaning
print("Cleaning text data...")
df['clean_text'] = df['text'].apply(clean_text)

# Check for empty texts after cleaning
empty_texts = df['clean_text'].apply(lambda x: x.strip() == "").sum()
print(f"✓ Cleaned {len(df)} texts ({empty_texts} texts were empty after cleaning)")

# Display sample of cleaned texts
print("\nSample of cleaned text (first 3 entries):")
print(df[['text', 'clean_text']].head(3).to_string())

# Preprocessing
print("\n4. TEXT PREPROCESSING")
print("-"*50)

# Define comprehensive preprocessing function
def preprocess_text(text, lemmatize=True, stem=False, remove_stopwords=True):
    """
    Comprehensive text preprocessing function that:
    - Tokenizes text
    - Removes stopwords (optional)
    - Applies lemmatization (optional)
    - Applies stemming (optional)
    """
    if not isinstance(text, str) or text.strip() == "":
        return []
    
    # Tokenize text
    tokens = tokenize_text(text)
    
    # Keep only alphabetic tokens and remove stopwords if requested
    if remove_stopwords:
        tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    else:
        tokens = [word for word in tokens if word.isalpha()]
    
    # Apply lemmatization if requested
    if lemmatize:
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Apply stemming if requested
    if stem:
        tokens = [stemmer.stem(word) for word in tokens]
    
    return tokens

# Apply standard preprocessing
print("Applying preprocessing...")
df['tokens'] = df['clean_text'].apply(lambda x: preprocess_text(x, lemmatize=True, stem=False, remove_stopwords=True))
df['preprocessed_text'] = df['tokens'].apply(lambda tokens: " ".join(tokens))
print("✓ Applied preprocessing")

# For comparison, also apply stemming-only preprocessing
print("Also applying stemming-only preprocessing for comparison...")
df['tokens_stemmed'] = df['clean_text'].apply(lambda x: preprocess_text(x, lemmatize=False, stem=True, remove_stopwords=True))
print("✓ Applied stemming-only preprocessing")

# Display sample of preprocessed texts
print("\nSample of preprocessed text (first 3 entries):")
print(df[['clean_text', 'preprocessed_text']].head(3).to_string())

# Analyze preprocessing results
print("\nAnalyzing preprocessing results...")
token_counts = df['tokens'].apply(len)
print(f"Average token count after preprocessing: {token_counts.mean():.2f}")
print(f"Min token count: {token_counts.min()}, Max token count: {token_counts.max()}")

# Dataset partitioning
print("\n5. DATASET PARTITIONING")
print("-"*50)

# Filter out entries with empty preprocessed text
df_filtered = df[df['preprocessed_text'].str.strip() != ""].copy()
print(f"Filtered dataset size: {len(df_filtered)} entries (removed {len(df) - len(df_filtered)} empty entries)")

# Create stratified train/val/test split (70/15/15)
X = df_filtered['preprocessed_text']
if 'emotion' in df_filtered.columns:
    y = df_filtered['emotion']
    # Check for NaN values in y and handle them
    if y.isna().any():
        print(f"Found {y.isna().sum()} NaN values in emotion labels")
        # Option 1: Fill NaN values with a default value
        y = y.fillna("unknown")
        print("✓ Filled NaN values with 'unknown'")
else:
    y = df_filtered['emotion_id']
    # Check for NaN values in y and handle them
    if y.isna().any():
        print(f"Found {y.isna().sum()} NaN values in emotion_id")
        # Use the most common emotion_id to fill NaN values
        most_common_id = y.mode()[0]
        y = y.fillna(most_common_id)
        print(f"✓ Filled NaN values with most common emotion_id: {most_common_id}")

# Create train/val+test split (70/30)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Create val/test split from the temp set (15/15 of original, which is 50/50 of the temp set)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

print(f"Training set: {len(X_train)} entries ({len(X_train) / len(df_filtered):.1%})")
print(f"Validation set: {len(X_val)} entries ({len(X_val) / len(df_filtered):.1%})")
print(f"Test set: {len(X_test)} entries ({len(X_test) / len(df_filtered):.1%})")

# Save the train/val/test sets
train_df = df_filtered.loc[X_train.index]
val_df = df_filtered.loc[X_val.index]
test_df = df_filtered.loc[X_test.index]

train_df.to_csv("processed_data/train.csv", index=False)
val_df.to_csv("processed_data/val.csv", index=False)
test_df.to_csv("processed_data/test.csv", index=False)
print("✓ Saved train/val/test sets to CSV files")

# Feature extraction
print("\n6. FEATURE EXTRACTION")
print("-"*50)

# TF-IDF Vectorization
print("Performing TF-IDF vectorization...")
tfidf = TfidfVectorizer(max_features=5000)
X_tfidf_train = tfidf.fit_transform(train_df['preprocessed_text'])
X_tfidf_val = tfidf.transform(val_df['preprocessed_text'])
X_tfidf_test = tfidf.transform(test_df['preprocessed_text'])

print(f"TF-IDF training feature matrix: {X_tfidf_train.shape}")
print(f"TF-IDF validation feature matrix: {X_tfidf_val.shape}")
print(f"TF-IDF test feature matrix: {X_tfidf_test.shape}")

# Save TF-IDF features
tfidf_data = {
    'vectorizer': tfidf,
    'X_train': X_tfidf_train,
    'X_val': X_tfidf_val,
    'X_test': X_tfidf_test,
    'y_train': y_train.values,
    'y_val': y_val.values,
    'y_test': y_test.values,
    'feature_names': tfidf.get_feature_names_out()
}

with open("processed_data/tfidf_features.pkl", "wb") as f:
    pickle.dump(tfidf_data, f)
print("✓ Saved TF-IDF features to 'processed_data/tfidf_features.pkl'")

# Transformer-Based Embeddings (optional - can be skipped if causing issues)
generate_transformer_embeddings = True  # Set to True to generate embeddings
print("\nGenerating transformer embeddings...")

if generate_transformer_embeddings:
    try:
        import torch
        from transformers import AutoTokenizer, AutoModel
        
        # Choose a transformer model
        model_name = "distilroberta-base"
        print(f"Loading {model_name} model...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
        
        def get_transformer_embeddings(text):
            if not isinstance(text, str) or text.strip() == "":
                # Return a zero vector of appropriate size if text is empty
                return torch.zeros(768).numpy()  # 768 is the dimension for distilroberta-base
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            # Average the token embeddings from the last hidden state
            embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            return embeddings
        
        # Process a small sample for demonstration
        sample_size = min(5, len(train_df))
        print(f"Generating transformer embeddings for {sample_size} sample training entries...")
        
        sample_train_df = train_df.head(sample_size).copy()
        sample_train_df['transformer_embeddings'] = sample_train_df['preprocessed_text'].apply(get_transformer_embeddings)
        
        # Print sample embeddings info
        print("\nTransformer Embeddings Information:")
        print(f"  Model: {model_name}")
        print(f"  Embedding dimension: {len(sample_train_df['transformer_embeddings'].iloc[0])}")
        print("\nSample embedding vectors (first entry, first 10 dimensions):")
        first_embedding = sample_train_df['transformer_embeddings'].iloc[0]
        print(f"  {first_embedding[:10]}")
        
        # Print the text and its embedding size for each sample
        print("\nSample texts and their embedding dimensions:")
        for i, row in sample_train_df.iterrows():
            embedding = row['transformer_embeddings']
            text = row['preprocessed_text']
            print(f"  Text: '{text[:50]}{'...' if len(text) > 50 else ''}' → Embedding shape: {embedding.shape}")
        
        # Save the sample embeddings
        transformer_data = {
            'embeddings': sample_train_df['transformer_embeddings'].tolist(),
            'texts': sample_train_df['preprocessed_text'].tolist(),
            'labels': sample_train_df['emotion_id'].tolist() if 'emotion_id' in sample_train_df.columns else None,
            'emotion_names': sample_train_df['emotion'].tolist() if 'emotion' in sample_train_df.columns else None
        }
        
        with open("processed_data/transformer_embeddings_sample.pkl", "wb") as f:
            pickle.dump(transformer_data, f)
        print("\n✓ Saved sample transformer embeddings to 'processed_data/transformer_embeddings_sample.pkl'")
        
    except Exception as e:
        print(f"! Error generating transformer embeddings: {e}")
        print("  Skipping transformer embeddings generation.")
else:
    print("Skipping transformer embeddings generation (disabled).")

# Visualization (optional - disabled by default)
generate_visualizations = True  # Set to True to enable visualizations
if generate_visualizations:
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from collections import Counter
        
        print("\n7. VISUALIZATION OF PREPROCESSING RESULTS")
        print("-"*50)
        
        # Create output directory for visualizations
        os.makedirs("processed_data/visualizations", exist_ok=True)
        
        # Visualize token count distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(token_counts, bins=50)
        plt.title('Distribution of Token Counts After Preprocessing')
        plt.xlabel('Number of Tokens')
        plt.ylabel('Frequency')
        plt.savefig("processed_data/visualizations/token_count_distribution.png")
        plt.close()
        
        # Visualize top words
        all_words = [word for tokens in df_filtered['tokens'] for word in tokens]
        word_freq = Counter(all_words)
        top_words = word_freq.most_common(20)
        
        plt.figure(figsize=(12, 6))
        sns.barplot(x=[word[0] for word in top_words], y=[word[1] for word in top_words])
        plt.title('Top 20 Words After Preprocessing')
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig("processed_data/visualizations/top_words.png")
        plt.close()
        
        # Visualize emotion distribution if available
        if 'emotion' in df_filtered.columns:
            plt.figure(figsize=(14, 7))
            emotion_counts = df_filtered['emotion'].value_counts()
            sns.barplot(x=emotion_counts.index, y=emotion_counts.values)
            plt.title('Emotion Distribution')
            plt.xlabel('Emotion')
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("processed_data/visualizations/emotion_distribution.png")
            plt.close()
        
        print("✓ Saved visualization charts to 'processed_data/visualizations' directory")
        
    except Exception as e:
        print(f"! Error generating visualizations: {e}")
else:
    print("\n7. VISUALIZATIONS (Skipped)")
    print("-"*50)
    print("Visualizations have been disabled to avoid potential issues.")

# Generate preprocessing report
print("\n8. GENERATING PREPROCESSING REPORT")
print("-"*50)

# Create word frequency list if not already created by visualization step
if 'word_freq' not in locals():
    all_words = [word for tokens in df_filtered['tokens'] for word in tokens]
    word_freq = {}
    for word in all_words:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1
    word_freq = {k: v for k, v in sorted(word_freq.items(), key=lambda item: item[1], reverse=True)}
    print("✓ Generated word frequency data for report")

report = {
    'dataset_size': len(df),
    'filtered_dataset_size': len(df_filtered),
    'train_size': len(train_df),
    'val_size': len(val_df),
    'test_size': len(test_df),
    'avg_token_count': float(token_counts.mean()),
    'min_token_count': int(token_counts.min()),
    'max_token_count': int(token_counts.max()),
    'tfidf_features': X_tfidf_train.shape[1],
    'transformer_model': model_name if 'model_name' in locals() and generate_transformer_embeddings else None,
    'embedding_dim': 768 if 'model_name' in locals() and generate_transformer_embeddings else None,
    'top_words': list(word_freq.items())[:100] if word_freq else None,
    'processing_time': time.time() - start_time
}

with open("processed_data/preprocessing_report.json", "w") as f:
    import json
    json.dump(report, f, indent=2)

# Save full preprocessed data
df_filtered.to_csv("processed_data/preprocessed_data_full.csv", index=False)
print("✓ Saved full preprocessed data to 'processed_data/preprocessed_data_full.csv'")
print("✓ Saved preprocessing report to 'processed_data/preprocessing_report.json'")

print("\n" + "="*80)
print(f"PREPROCESSING COMPLETED IN {time.time() - start_time:.2f} SECONDS")
print("="*80)
