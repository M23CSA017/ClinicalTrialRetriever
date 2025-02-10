# # src/similarity.py

import logging
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .utils import sentence_based_chunking


import logging
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from datasets import Dataset

logger = logging.getLogger(__name__)

def compute_similarity_dataset(anchor_df, model, top_k=10):
    """
    Compute similarity pairs and create a dataset for Sentence Transformers Trainer.
    
    Args:
        anchor_df (pd.DataFrame): DataFrame with cluster labels and text chunks
        model (SentenceTransformer): Embedding model
        top_k (int): Number of top similar pairs to select per anchor
    
    Returns:
        Dataset: Prepared dataset with sentence pairs and similarity scores
    """
    pair_data = {
        'sentence1': [],
        'sentence2': [],
        'score': []
    }

    # Group anchors by cluster
    grouped = anchor_df.groupby("cluster_labels")

    for cluster_id, cluster_data in grouped:
        if cluster_id == -1:
            continue

        # Split cluster into 40% anchors and 60% candidates
        anchors = cluster_data.sample(frac=0.4, random_state=42)
        candidates = cluster_data.drop(anchors.index)

        # Encode candidate texts
        candidate_texts = candidates["chunk_text"].tolist()
        candidate_embs = model.encode(candidate_texts, convert_to_numpy=True)

        # Process each anchor
        for _, anchor_row in anchors.iterrows():
            anchor_text = anchor_row["chunk_text"]

            # Encode the anchor text
            anchor_vec = model.encode([anchor_text], convert_to_numpy=True)[0]

            # Compute cosine similarities
            sim_scores = cosine_similarity([anchor_vec], candidate_embs)[0]

            # Get top_k indices
            top_indices = np.argsort(sim_scores)[::-1][:top_k]

            # Add top pairs to dataset
            for i in top_indices:
                pair_data['sentence1'].append(anchor_text)
                pair_data['sentence2'].append(candidate_texts[i])
                
                # Use cosine similarity as score (normalized between 0 and 1)
                pair_data['score'].append(float(sim_scores[i]))

    logger.info(f"Created dataset with {len(pair_data['sentence1'])} pairs")
    
    # Convert to Hugging Face Dataset
    return Dataset.from_dict(pair_data)

def prepare_training_datasets(anchor_df, model, train_split=0.8, top_k=10):


    # Create full similarity dataset
    full_dataset = compute_similarity_dataset(anchor_df, model, top_k)
    
    # Split into train and validation
    train_size = int(len(full_dataset) * train_split)
    train_dataset = full_dataset.select(range(train_size))
    eval_dataset = full_dataset.select(range(train_size, len(full_dataset)))
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset