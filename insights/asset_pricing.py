# Import necessary libraries
import os
import json
import nltk
import numpy as np
from tqdm import tqdm
from nltk import sent_tokenize, word_tokenize
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Ensure required NLTK data is downloaded
nltk.download('punkt')

# Load the SentenceTransformer model
model_name = 'all-mpnet-base-v2'  # You can choose a different pre-trained model if desired
embedder = SentenceTransformer(model_name)

# Function to read and preprocess text into paragraphs
def read_and_preprocess(file_path, min_paragraph_length=75):
    print("Reading and preprocessing text...")
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    # Split text into initial paragraphs based on double newlines
    paragraphs = text.split('\n\n')
    # Clean paragraphs and ensure they have at least min_paragraph_length words
    processed_paragraphs = []
    for para in paragraphs:
        # Remove extra whitespace
        para = para.strip()
        # Remove empty paragraphs
        if not para:
            continue
        # Tokenize paragraph into words
        words = word_tokenize(para)
        if len(words) >= min_paragraph_length:
            processed_paragraphs.append(para)
    return processed_paragraphs

# Function to embed paragraphs in batches of 10
def embed_paragraphs(paragraphs, batch_size=10):
    print("Embedding paragraphs...")
    embeddings = []
    for i in tqdm(range(0, len(paragraphs), batch_size)):
        batch_paragraphs = paragraphs[i:i+batch_size]
        batch_embeddings = embedder.encode(batch_paragraphs, convert_to_tensor=True, normalize_embeddings=True)
        embeddings.extend(batch_embeddings)
    embeddings = np.array([embedding.cpu().numpy() for embedding in embeddings])
    return embeddings

# Function to combine paragraphs with over 95% similarity
def combine_similar_paragraphs(paragraphs, embeddings, similarity_threshold=0.95):
    print("Combining similar paragraphs...")
    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings)
    # Keep track of which paragraphs have been merged
    merged_indices = set()
    combined_paragraphs = []
    combined_embeddings = []
    for i in range(len(paragraphs)):
        if i in merged_indices:
            continue
        similar_indices = [j for j in range(len(paragraphs)) if similarity_matrix[i][j] >= similarity_threshold and j != i]
        # Merge paragraphs
        combined_text = paragraphs[i]
        for j in similar_indices:
            if j not in merged_indices:
                combined_text += ' ' + paragraphs[j]
                merged_indices.add(j)
        combined_paragraphs.append(combined_text)
        combined_embeddings.append(embeddings[i])
        merged_indices.add(i)
    combined_embeddings = np.array(combined_embeddings)
    return combined_paragraphs, combined_embeddings

# Function to cluster paragraphs into topics using K-Means
def cluster_paragraphs(embeddings, num_clusters=10):
    print("Clustering paragraphs into topics...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(embeddings)
    return cluster_labels, kmeans

# Function to remove far outliers
def remove_outliers(paragraphs, embeddings, cluster_labels, kmeans, threshold=2.0):
    print("Removing outliers...")
    # Compute distances to cluster centers
    distances = kmeans.transform(embeddings)
    min_distances = np.min(distances, axis=1)
    mean_distance = np.mean(min_distances)
    std_distance = np.std(min_distances)
    # Identify outliers
    outlier_indices = [i for i in range(len(paragraphs)) if (min_distances[i] - mean_distance) > threshold * std_distance]
    # Remove outliers
    filtered_paragraphs = [para for i, para in enumerate(paragraphs) if i not in outlier_indices]
    filtered_embeddings = np.array([embeddings[i] for i in range(len(paragraphs)) if i not in outlier_indices])
    filtered_labels = np.array([cluster_labels[i] for i in range(len(paragraphs)) if i not in outlier_indices])
    return filtered_paragraphs, filtered_embeddings, filtered_labels

# Function to clean text (basic cleanup)
def clean_text(text):
    # Remove non-alphanumeric characters except for basic punctuation
    cleaned_text = ''.join(c if c.isalnum() or c.isspace() or c in '.,;:?!' else ' ' for c in text)
    # Remove extra whitespace
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text

# Function to organize paragraphs into topics and save to JSON
def save_paragraphs_to_json(paragraphs, cluster_labels, output_file='categorized_paragraphs.json'):
    print(f"Saving paragraphs to {output_file}...")
    categorized_paragraphs = {}
    for i, label in enumerate(cluster_labels):
        topic_key = f"Topic {label + 1}"
        if topic_key not in categorized_paragraphs:
            categorized_paragraphs[topic_key] = []
        cleaned_para = clean_text(paragraphs[i])
        categorized_paragraphs[topic_key].append(cleaned_para)
    # Save to JSON
    with open(output_file, 'w', encoding='utf-8') as json_file:
        json.dump(categorized_paragraphs, json_file, indent=4)
    print(f"Paragraphs saved to {output_file}")

# Main function to execute the process
def process_text(file_path):
    # Step 1: Read and preprocess text into paragraphs
    paragraphs = read_and_preprocess(file_path)

    # Step 2: Embed paragraphs in batches of 10
    embeddings = embed_paragraphs(paragraphs, batch_size=10)

    # Step 3: Combine similar paragraphs with over 95% similarity
    combined_paragraphs, combined_embeddings = combine_similar_paragraphs(paragraphs, embeddings, similarity_threshold=0.95)

    # Step 4: Cluster paragraphs into 10 topics using K-Means
    cluster_labels, kmeans_model = cluster_paragraphs(combined_embeddings, num_clusters=10)

    # Step 5: Remove far outliers
    filtered_paragraphs, filtered_embeddings, filtered_labels = remove_outliers(combined_paragraphs, combined_embeddings, cluster_labels, kmeans_model, threshold=2.0)

    # Step 6: Save the results to JSON
    save_paragraphs_to_json(filtered_paragraphs, filtered_labels)

# Provide the path to your text file
file_path = 'cochrane_asset_pricing.txt'

# Execute the processing
process_text(file_path)